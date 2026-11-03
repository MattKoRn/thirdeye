# Live control channel — agent-driven play & debugging

Status: **phases 1+2 implemented** ([control.cpp](../apps/thirdeye/control.cpp),
2026-07-19); phase 3 (Burial Glen POC) and phase 4 (skill) pending. This doc is
the hand-off spec: any agent (or human) should be able to implement a phase
from it without re-deriving context. Target: an interactive protocol for
driving a running thirdeye — inject input, query game state, snapshot the
screen — culminating in an agent playing the Burial Glen end-to-end and saving.

## Motivation

Today's automation is **fire-and-forget**: `THIRDEYE_AUTOWALK` plays a fixed
token script decided before launch; `THIRDEYE_DUMP` snapshots every present;
diagnosis happens post-mortem from BMPs and logs. That's fine for regression
walks but useless for anything *reactive* — combat (where's the mist now?),
exploration (what did that chest drop?), or interactive debugging ("click
that button, now show me the cell state — without restarting").

The fix is a live, bidirectional control channel. Two capabilities:

1. **Act**: inject keys/clicks into the running game, exactly as AUTOWALK
   does internally, but on demand instead of on a fixed cadence.
2. **Observe**: query the *object system* directly — party pose, monster HP,
   floor items, cell contents. This is strictly better than OCR'ing
   screenshots: we own the VM, so we can read the same statics the SOP
   reads. Screenshots stay available for visual confirmation.

An agent (Claude via Bash, a Python script, a human with `socat`) then runs a
sense→decide→act loop against a live game. The same channel doubles as an
interactive debugger.

## What already exists (reuse, don't reinvent)

All in [engine.cpp](../apps/thirdeye/engine.cpp)'s `pumpHost` region
(~line 243 onward) — the host seam called from every `dispatch_event` /
`peek_event` ([runtime/internal.hpp](../apps/thirdeye/runtime/internal.hpp)):

- **Key injection**: `events.postEvent(0, VM::SYS_KEYDOWN, scancode)` — how
  AUTOWALK `Key` tokens fire (engine.cpp ~457).
- **Click injection**: `gfx.mouseToLogical(x, y, lx, ly)` →
  `events.mouseMove(lx, ly)` → `events.mouseButton(left, right)`, with the
  **release ~5 pumps later** (`events.mouseButton(false, false)`) — see the
  AUTOWALK `Lclick`/`Rclick` phase-5 release (engine.cpp ~474). The delay
  matters: the SOP polls button state across pumps; an instant release is
  missed.
- **Automap toggle**: host-side only — `THIRDEYE::automap::toggle()`
  (a VM SYS_KEYDOWN can't reach it).
- **Screenshot**: `Graphics::saveScreenshot(path)` (what THIRDEYE_DUMP
  calls, graphics.cpp ~1092).
- **Object reads**: `automap.cpp` has the read patterns to copy —
  `lvlobjHead()` (automap.cpp:132) and `featureDisabled()` (automap.cpp:171).

## Design

### Transport

- Env var `THIRDEYE_CTL=<path>` → engine creates a **Unix domain socket**
  (stream) at `<path>` at boot, `unlink`s any stale file first, sets
  `O_NONBLOCK` on the listen fd.
- **No threads.** Polled entirely from `pumpHost`, once per pump (~30 Hz):
  non-blocking `accept()`, non-blocking `read()` of complete lines,
  handlers run synchronously, replies `write()`n back. One client at a
  time is fine (reject or replace on second accept — implementer's choice,
  document which).
- Not set ⇒ zero cost, zero code active. macOS + Linux first; Windows
  (AF_UNIX exists on Win10+, else named pipes) explicitly out of scope v1.
- Latency budget: a command waits at most one pump (~33 ms). Fine.

### Protocol

Line-oriented text, UTF-8, `\n`-terminated. One command per line. Every
command produces one reply block terminated by a line reading `ok` or
`err <message>`. Data lines precede the terminator. No JSON — parseable
with awk/grep, writable without a library.

```
key <hex>                inject SYS_KEYDOWN (4800=fwd 5000=back 4b00/4d00=strafe
                         4700/4900=turn 0d=Enter 1b=Esc — AUTOWALK codes)
click <L|R> <x> <y>      press at logical 320x200 coords; auto-release ~5 pumps
                         later (queued by the handler, no client action needed)
map                      toggle the automap overlay
dump <path>              write a BMP of the current frame to <path>

party                    party pose + members
cell <x> <y>             lvlobj chain heads at that cell, all 3 planes
items                    floor items (W:place == -1) on the party's level
monsters                 live NPCs on the party's level
obj <id>                 class + entity fields of one object (cls/place/x/y/
                         lvl/region/next/prev) — the per-object complement to
                         the list commands; lets a client scan slots
peek <obj> <off> <n>     hex dump of an object's statics (debug)
send <obj> <msg> [args]  SEND a message to an object (debug, dangerous — it
                         runs SOP bytecode; document as "you can corrupt state")
save <slot>              run the save flow into slot N (phase 2+; may start as
                         err unimplemented)
ping                     liveness check, replies ok immediately
```

Reply sketches (stable, documented formats — agents will parse these):

```
> party
pose x=15 y=10 fdir=3 lvl=1
pc slot=0 obj=32 name=Sir_Mikeal hp=34/40 status=0x00
pc slot=1 obj=33 name=Stonebeard hp=7/45 status=0x00
ok

> cell 14 10
plane0 head=1050 chain=1050:2208,1268:2268
plane1 head=-1
plane2 head=-1
ok

> items
item obj=103 cls=1376 x=21 y=8 region=2
ok

> monsters
npc obj=1764 cls=1904 x=15 y=9 hp=12
ok
```

Naming: `chain=` entries are `slot:class` pairs in chain order. Spaces in
names become `_`. Numbers decimal except `peek` output and `status` bits.

### Query internals (the offsets an implementer needs)

All class numbers / offsets verified in this codebase — grep for them in
[savegame/lvl_tmp.cpp](../apps/thirdeye/savegame/lvl_tmp.cpp),
[automap.cpp](../apps/thirdeye/automap.cpp), [runtime/eye.cpp](../apps/thirdeye/runtime/eye.cpp):

| What | How |
|---|---|
| kernel object | `objects.firstObjectOfClass(1382)` |
| party x/y/fdir/lvl | kernel class-1382 bytes @243/244/245/246 |
| W:in_hand | kernel @241 (u16, -1 = empty hand) |
| W:current_screen | kernel @265 (also `uiScreenActive()` in internal.hpp) |
| player[] slots | see `kPlayerSlots` handling in runtime/eye.cpp resume_level |
| PC class | 1369; name @137 (20 B), hpts @171, hmax @173, PCstat @161 |
| dungeon object | `objects.firstObjectOfClass(1381)` |
| lvlobj cell | dungeon class-1381 @ `1024 + plane*2048 + y*64 + x*2` (u16 head, 0xFFFF = empty; planes: 0=features, 1=items, 2=monsters) |
| entities statics | class 1370: W:place@0, B:x@2, B:y@3, B:lvl@4, B:region@5, W:next@6, W:prev@8 |
| monster? | `objects.isSubclassOf(cls, 1622)` (NPC) |
| item? | `objects.isSubclassOf(cls, 1371)` (items) |
| feature disabled (tree cut, door open) | features class-1994 @0 bit 0x8000 (`featureDisabled`, automap.cpp) |
| NPC hitpts | class-1622 block @3, u16 (verified via `daesop -k EYE.RES 1622`: `W:hitpts` = PUBLIC_STATIC_VARIABLE 3) |

**Phase-3 trophy:** `items` returning empty at QSP boot was not a spawn
mechanism mystery — it exposed a real savegame bug: `parseItemStream` misread
the 8-byte CDESC record header as 4 bytes, shifting every item's statics and
discarding the placement fields. Fixed 2026-07-19 (see
[eob3_savegame_format.md](eob3_savegame_format.md) §2.3); all 26 Burial Glen
floor items (wands @ (27,30), arrows/bow/mail @ (2,10), the axe pile @ (12,2))
now exist, link into lvlobj plane 1, render in the 3D view, and list via
`items` — matching GameBanshee's legend annotation-for-annotation.

Chain walks must carry a guard (≤ 2000 hops) — see the invariant checker in
lvl_tmp.cpp for the pattern; a corrupt chain must produce `err`, not a hang.

### Files & integration

- New: `apps/thirdeye/control.{cpp,hpp}` — socket lifecycle, line buffering,
  command dispatch. Public surface kept minimal:
  `control::init(path)`, `control::pump(ControlDeps&)`, `control::shutdown()`.
  `ControlDeps` = `{Graphics*, EventSystem&, ObjectSystem&, Resource&}` —
  same references `pumpHost` already holds; gfx nullable for headless.
- Hook: one call in `pumpHost` next to the AUTOWALK block, gated on the env
  var having been set. The deferred click-release is a tick counter inside
  control.cpp, mirroring AUTOWALK's phase-5 logic.
- CLI: env var only for now (`--ctl=<path>` skipped until someone asks —
  every driver script sets env vars anyway for MUTE/DUMP).

## Phases

**Phase 1 — channel + act + dump.** Socket, `ping`/`key`/`click`/`map`/
`dump`. Acceptance: from Bash, boot with `THIRDEYE_CTL`, walk a corridor by
sending `key 4800`s, `dump` before/after, pixels differ; a second client
connect behaves as documented; killing the client mid-command doesn't wedge
the pump (write must tolerate EPIPE).

**Phase 2 — observe.** `party`/`cell`/`items`/`monsters`/`peek`/`send`.
Acceptance: on `--load-save` into a known slot, `party` matches the save's
ITEMS position record; `cell` on a known door cell lists the door+button
chain (cf. the chain-order work in lvl_tmp.cpp); `items` on LVL03 lists the
wands the GameBanshee legend places at its annotation #2
(`../eob3_research/gamebanshee/legends/burialglen.json`).

**Phase 3 — the Burial Glen POC.** In progress (2026-07-19);
[scripts/glen_drive.py](../scripts/glen_drive.py) +
[scripts/ctl.py](../scripts/ctl.py) are the driver. Coordinates harvested so
far (logical 320x200, right-click to attack): PC0 weapon (229,20), PC1
(296,18), PC3 (297,70); CAMP button ≈(296,183). Note the engine.cpp AUTOWALK
example's "R126:133 = PC0's right hand" lands inside the movement arrow pad,
NOT a weapon icon — use the panel coordinates above. Facing: fdir 0=N 1=E
2=S 3=W; dead monsters read x=255 y=255 hp<=0 (filter them client-side).
**ALL ATTACK** (the big combat lever, per GameBanshee gameplaynotes.html):
left-click each PC's name plate to toggle it yellow — plates at (216,6),
(286,6), (216,57), (286,57), (216,110), (286,110) — and with ≥1 selected an
ALL ATTACK button appears under the arrow pad at ≈(146,165); one left-click
there swings every selected PC. ~2× the kill rate of clicking weapon icons,
and the selection persists. The plain space bar (`key 20`) does nothing.
An agent drives, via the channel only:
QSP start on LVL03 → kill the grave mists (query `monsters`, face, attack
via weapon-icon clicks, re-query until dead) → pick up loot at the legend's
annotations #2–#4 → find the axe, equip it, chop trees toward the gate
(verify with `featureDisabled` via `cell`) → reach exit B (Forest Trail
gate) → camp → save to a slot → verify the slot's `.BIN`s exist and parse.
The driving script (Python, lives in `scripts/`) is throwaway-quality; the
*channel* is the deliverable. Expect this phase to surface runtime bugs
(axe equip, chop mechanics) — that's a feature; file/fix them as found.

**Phase 4 — skill.** `.claude/skills/thirdeye-drive/SKILL.md` so any future
session can drive a live game without rediscovering the protocol. Contents:
how to boot with the channel (always `THIRDEYE_MUTE=1`), the command table,
reply grammar, the scan-code crib, known screen coordinates (weapon icons,
CAMP button, compass arrows — harvest these during phase 3), and the
sense→decide→act loop pattern with the "one command may take a pump" caveat.
Trigger phrases: "drive the game", "play the glen", "live debug", "control
channel".

## Driving scripts (the "sweeper")

[scripts/ctl.py](../scripts/ctl.py) is the one-shot client;
[scripts/glen_drive.py](../scripts/glen_drive.py) layers level-agnostic
primitives on it. Everything reads live state (lvlmap, cell chains, party
pose), so **any save on any level works** — play to wherever, save, quit,
then re-drive that save headless:

```bash
# 1. boot the save you want to explore (slot N), with the channel armed
THIRDEYE_CTL=/tmp/te_ctl.sock THIRDEYE_MUTE=1 \
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  build/thirdeye.app/Contents/MacOS/thirdeye ../data/EYE.RES \
  --skip-intro --skip-menu --load-save N --nosound --scale=1 &

# 2. drive it
python3 scripts/ctl.py /tmp/te_ctl.sock party items monsters  # look around
python3 scripts/glen_drive.py /tmp/te_ctl.sock scan           # walkability map
python3 scripts/glen_drive.py /tmp/te_ctl.sock goto 14 10     # BFS-walk (fights,
                                                              # chops, cutscenes)
python3 scripts/glen_drive.py /tmp/te_ctl.sock loot 14 10     # pick up a cell's
                                                              # floor items
python3 scripts/glen_drive.py /tmp/te_ctl.sock kill-adjacent  # ALL-ATTACK front
```

Caveats for non-Glen levels: the walkable/hackable plane-0 class sets at the
top of glen_drive.py are graveyard classes (graves 2054 walkable, trees 2060
choppable) — indoor levels' doors/levers/pits need their own entries; `loot`
requires standing ON the item cell (ahead-cell clicks lose to feature click
regions — fruit trees ate ours); and `heal_party` is a debug poke, remove it
for honest runs. The per-run orchestration (which cells to visit) is a
5-line script over `goto`/`loot` — see the phase-3 sweep scripts for the
pattern.

## Testing

- Protocol unit tests (apps/tests): `control_tests.cpp` covers the pure
  `tokenize()` (fragmented/joined lines are the socket layer's job — it
  splits on `\n` before tokenize sees anything).
- Headless e2e: `scripts/ctl_e2e.py`, registered as ctest `control_e2e` —
  boots with `THIRDEYE_CTL` + `--load-save 1`, scripts
  `ping`→`party`→`key 4900`→`party`, asserts the pose changed. Skips
  (exit 77) unless `THIRDEYE_TEST_DATA_DIR` points at an EOB3 install.
- The valgrind walk (`scripts/ci-valgrind.sh`) must stay green with the
  channel compiled in but env-var-off (zero-cost path).

## Non-goals (v1)

- Windows transport, multiple simultaneous clients, authentication (it's a
  local debug socket; document that it's as trusted as ptrace).
- Frame-perfect input timing (one-pump granularity is enough for EOB3).
- A general RPC layer — this is a debug/drive channel, not an API contract;
  the format may change between versions without ceremony.

## Open questions (answered during phases 1–2)

1. Second client: **rejected with `err busy`** and dropped. Implementation
   note: the pump drains the existing client's recv (detecting EOF) *before*
   accepting — otherwise a fast connect/close/connect (a scripted
   one-command-per-connection client) sees the stale fd and wrongly gets
   `err busy`.
2. `save <slot>`: **stubbed as `err unimplemented`**; phase 3 drives the camp
   flow via clicks first, per the lean.
3. `dump`: **path-only**; the agent reads the BMP itself.


## Neural in-game rolling saves and game-over recovery

When the explicitly requested **`nn auto on`** mode is enabled in **Eye of the Beholder 3**, the host
calls the game's existing native `save_game(slot, level)` routine every
60 seconds of monotonic elapsed time (rather than counting frames). Saves
wait for the actual dungeon view, a valid current level, and at least one
living party member. A long UI/menu pause does not trigger a burst of saves.
The three rotating backups are named **nn-1**, **nn-2**, **nn-3**, backed by
native EOB3 slots **12, 13, 14** (`SAVEGAME/ITEMS_12.BIN`, `LVLxx_12.BIN`,
and similarly 13/14). These slots are outside the normal 12-slot picker, so
ordinary player save slots 0..11 are never overwritten. Names appear in
`SAVEGAME.DIR` entries 12..14 and survive a regular manual save.

On a game-over detected from **all registered party members having zero or
negative HP**, a five-second timer starts; disappearance of the kernel during
the death screen does not cancel it. If a party member becomes alive again,
the countdown is cancelled. At five seconds the host selects the newest
named, structurally present backup containing a **living** party, restores
the native ITEMS/LVL backup files and automap, resets SOP VM instances and
event windows, and resumes the normal EOB3 continue-game boot, leaving neural
AUTO enabled. If no healthy NN save exists, recovery is not attempted and a
single diagnostic is emitted. Corrupt/missing/unrelated slots are never
promoted to a recommended save. Console logs show successful `nn-N`
rotations and recovery.

**Current scope:** EOB3 only. Dungeon Hack uses a different savegame
implementation; the native EOB3 save/restore format must not be applied to
`HACK.RES`. Also, a scripted game-over with living party HP is not detected
by this first version. The three named saves are real **game-state** saves,
separate from neural policy checkpoints in `Documents/ThirdeyeNN`.

## Self-learning neural movement and F1 advice

The neural controller does **not** learn from the player's movement keys or
from `key` control commands. Press **F1** during gameplay to open a ranked
advice menu instead of autonomous takeover: choose one of up to five actions
to execute and learn from its measured outcome before the next batch appears.
For fully automatic play, the separate control command `nn auto on` remains
available; `nn auto off` returns control. Menus and other UI screens suspend
autonomous movement.

The controller is a lightweight online softmax policy with six actions:
forward (4800), backward (5000), strafe left/right (4b00/4d00), and
turn left/right (4700/4900). Each action is selected by the policy
using decaying epsilon-greedy exploration. The next host update compares
actual position and facing with the state before the action. It learns
from **its own successes and failures**, without demonstrations: first-time
cell visits receive a larger reward, revisits and turns a small reward,
and moves that fail receive a penalty. Known walls are excluded from
choices. Level-aware visit memory prioritizes accessible unexplored cells,
then falls back to learned choices for backtracking and turns. Repeating
failed actions at the same pose is suppressed until position or facing
changes. Same-floor teleports, multi-level scripted resets, and UI
transitions are excluded from feedback; adjacent-level movement transitions
now earn an exploration reward. The online reinforcement learning rate
gradually decays from 0.10 toward 0.025 and resumes from the saved training
count after restart. Each trained action logs its outcome, reward, applied
learning rate, sample count, and checkpoint status; successful automatic
checkpoints are logged too. The socket's `nn status` command reports sample
count, mean reward, success ratio, learning rate, checkpoint state
(new/pending/saved/unavailable), and training enabled/disabled state.
Importing a checkpoint while autonomous play is active discards stale
feedback from the old policy before applying the loaded weights.

**Enemy-specific tactical experiments and persistent weaknesses.**
The EOB3 neural controller now inspects the **equipped right-hand item**
of each living candidate attacker and distinguishes a weapon from an in-hand
spell scroll or wand. For a verified NPC in the forward cell, it first
experiments with untried attack/item types, then uses observed **damage
per attempt** plus an exploration bonus to choose tactics previously effective
against that NPC's **SOP class**. The experiment key records enemy class,
equipped item class, spell-like kind, and the spell identifier if that
identifier is explicitly exported by the item class; otherwise spell ID 0
means not verified. Combat actions use the actual right-click item event
and measure NPC HP afterward, allowing up to five action ticks for delayed
spell/projectile results. For magic-item actions, an observed HP decrease
confirms damage. If HP is unchanged, a **confirmed spent scroll or decrease
in a named exported wand charge counter** is required before recording a
zero-damage experiment. This is evidence of a non-damaging attempt, **not**
proof of immunity (it could be a miss or a non-damaging effect). If neither
damage nor resource use can be observed, the attempt is unverified, is not
trained as resistance, and that tactic is suppressed for the current monster
and pose so combat can try another available action. Known-empty wands are
excluded; unknown charges never default to zero. Invalid NPC readings,
changed party pose, or interrupted menus are excluded from feedback.

Per-enemy experience automatically saves to
`Documents/ThirdeyeNN/weaknesses.ckpt` every eight confirmed attacks and
on shutdown. It is a new checksum-validated sidecar (format
`THIRDEYE_TACTICS 1`) that stores trials, hits and cumulative damage by
enemy and tactic. A failed load never clears the in-memory data, and an
invalid existing file is not overwritten. Existing `brain.ckpt` movement
weights and `combat.ckpt` combat policy weights are unchanged.
`nn status` reports `tactic_samples`, `enemy_tactics`, and
`tactic_checkpoint`. The console reports each experiment and the current
running damage average.

**Limits:** Spell scrolls are consumed when used; the agent does not yet
restock them. Runtime test rails verify that spellbook/holy-symbol activation
uses magic M:29 and that direct magic M:54 reaches part of the cast chain,
but the M:54 synthetic harness with sp=0 did not confirm target damage
(`docs/progress.md`); neither identifies a safe memorized-slot dispatch
key or a complete, verified menu-selection path. The autonomous controller
therefore does **not** invent spell-menu coordinates or cast arbitrary
memorized spells. This iteration makes supported equipped wand/scroll spells
resource-aware and learns credible zero-damage experiments; full book/symbol
casting, non-damaging status effects and spell replenishment remain future
work. Spell IDs are separated only when exposed by actual class exports.
As before, this combat path is EOB3-only; Dungeon Hack has different VM
class layouts and UI coordinates.

**Verified native spell-menu probe (opt-in, no speculative casts).**
The control channel now exposes `nn spell status <slot>` (party slot 0..5),
reporting that character's raw-runtime-derived mage/cleric **prepared-copy
counts**, four-byte magic-effect bitmask and native `W:current_screen`.
The count calculation follows EOB3 `spell_list` in `runtime/eye.cpp`:
mage offsets 10..99 and cleric offsets 110..199 of the PC's
`B:spell_cnt[200]` at class-static offset 209. These counters **do not**
provide spell-class IDs or valid arguments for `magic.cast_spell` M:54.
Unsupported/invalid counters are rejected instead of treated as castable.

After `nn auto on`, `nn spell probe <slot> mage|cleric` queues one native
`magic.activate` M:29 for an alive PC with prepared copies. On the next
host pump it uses the same `magic.deactivate` M:172 verified by
`THIRDEYE_SPELLMENU` in `runtime/event.cpp`, then reports the actual
menu screen before/after, prepared copies and magic effects. It never sends
menu coordinates, selection keys or M:54. Explicit `nn auto on` mode pauses its normal
movement/combat actions for the probe; closing is attempted even if the
user disables autonomy or a menu is active. `nn status` includes
`spell_probe=queued|open|menu-cycle-verified|menu-transition-unverified|failed`.
While explicit `nn auto on` autonomy is active, the controller also reports observed changes
in real spell counters/effects as `[nn] spell-state`, clearly marked
`attributed_cast=no trained=no`. This adds live discovery telemetry
without pretending that a resource change proves an enemy weakness.
Neither `brain.ckpt`, `combat.ckpt`, `weaknesses.ckpt`, nor game-save
formats are altered; `nn-1`/`nn-2`/`nn-3` recovery is unchanged.

**Remaining limitation:** Spellbook/holy-symbol spell selection and the
M:54 `sp` dispatch mapping still require verified event/bytecode tracing.
A menu-open/close probe is not an autonomous spell cast; the neural agent
does not use unverified input mappings or train on these observations.

**F1 neural advice menu: true game pause, available in both games.**
Press **F1** while playing **Eye of the Beholder 3 or Dungeon Hack** to
open an in-game overlay of at most **five clickable suggestions**. The
host then stays inside a modal SDL event loop: only F1, Escape, numbered
choices, mouse hover/click, display/fullscreen changes and window-close are
accepted. **The SOP VM does not dispatch any game events while this panel
is open.** The heartbeat also subtracts the exact paused wall-clock
duration to prevent monsters, timers and spells from catching up after the
pause ends. F1 / Escape exits the pause without taking an action. Each recommendation is a real supported movement
action (forward, back, strafe left/right, turn left/right), ranked using the
currently saved `brain.ckpt` neural weights, accessible paths, nearby walls,
and exploration of unvisited cells. The display is ranked and bounded even
when all six movement actions are legal; inaccessible actions are excluded.
The advice comes from the game's trained neural policy, not a language-model
API and not random prose. For **Dungeon Hack**, the existing native
`draw_walls(x,y,facing,...,lvlmap,...)` call publishes a validated, live
32×32 maze and party pose. The same movement actions, rankings, outcome
feedback and five-choice menu apply without reading EOB3-only SOP class
statics. Dungeon Hack learns into a distinct checksum-protected
`Documents/ThirdeyeNN/brain-dungeon-hack.ckpt`; EOB3 keeps its unchanged
`brain.ckpt`.

Click a numbered row or press **1–5** to execute exactly one recommended
game action. The overlay closes for that action. After the game's SOP engine
has consumed the native movement key and the next state can be observed,
the controller compares the actual position/direction and computes the
existing neural exploration reward. Confirmed success or failure updates
`brain.ckpt` using the same online reinforcement logic as the original
autonomous mode. Invalid teleport, level transition or UI interruption
does **not** train the policy. It then **always returns to the F1 advice panel** with a fresh batch
of up to five suggestions, including the latest learning count and result.
When a game scene changes, an input times out, or the player reaches a
non-playable UI, the panel still reappears but shows an empty, safe result
instead of inventing actions. Close it with F1 / Escape to resume the
underlying game screen.
Press **F1** or **Escape** to close the menu; F1 during an outstanding action
prevents the next menu from automatically reopening. The overlay consumes
its own mouse clicks so no underlying inventory or combat control also fires.

**Mouse takeover removed:** The operating-system pointer is no longer
warped or hijacked. Regular motion and clicks always pass to the game when
the F1 advice overlay is closed; while it is open, they only interact with
the overlay. The human-like cursor driver, automatic UI navigation and
unattended food clicking were removed. The separately available
`nn food use <slot>` remains a deliberate **explicit** action against a
previously verified native food region. An optional advanced socket command
`nn auto on` still enables the existing autonomous movement/combat agent,
but does **not** take ownership of the player's mouse. F1 itself now opens
the advice menu instead of switching continuous AUTO on/off.

For control clients, use `nn advice open`, `nn advice list`,
`nn advice choose 1` through `nn advice choose 5`,
`nn advice close` and `nn advice status`. The normal `nn status`
reports `advice_open`, `advice_generation`, and `advice_choices`.
The policy checkpoint schema, survival/combat/food/tactics files, and
one-minute rotating EOB3 `nn-1`/`nn-2`/`nn-3` native saves with
five-second game-over recovery are unchanged. **Dungeon Hack's
native save/restore, monster HP, attack hit regions, food and spell
statics are different from EOB3's.** Those systems remain EOB3-only until
their own runtime controls and outcomes are verified; the same neural
advice and movement-learning interface is now active in both games without
unsafe cross-game commands.

**Native food-region calibration and explicit guarded use (EOB3 only).**
The neural agent now records the exact **registered SOP mouse-region
listener** responsible for a successful native food-use experiment. The
VM event layer captures real left-click coordinates, the live window's
rectangle, its event type, handler class/message, and handler object. The
capture rejects ambiguous smallest regions, full-screen regions and
unregistered clicks; a manual trial must contain exactly one observed
registered click, while the native UI is up, followed by both the
`W:in_hand` item disappearing and the targeted PC's food counter increasing.
No reward or actuator calibration comes from a click without that evidence.

To calibrate: turn neural autonomy/training on, hold a documented food item
in the native cursor, **open the intended character's equipment/food UI**,
run `nn food probe <slot>`, and click the food plate **once**. On verified
consumption, the agent records the observed region for that party slot,
food item class and `W:current_screen`. An observed handler belonging to a
different object is kept as diagnostic evidence only, not granted replay
privileges. Calibrations persist in a separate
`Documents/ThirdeyeNN/food_actuator.ckpt` file with version-1 header,
checksum, strict region validation, bounded entries, atomic replacement and
validate-before-commit load. A damaged file is preserved and automation
disabled until repaired; older neural checkpoints remain compatible.

With another recognized food item *already held*, open the intended
character's equipment/food screen and issue `nn food use <slot>`.
Only if a saved calibration matches the exact current party slot, item
class, screen, original PC object, **PC-class SOP region handler**,
handler message and window rectangle does the controller issue one
native left-click at the previously observed position, only when explicitly
requested via `nn food use`. It releases the
button on the next host pump and independently verifies the actual food
increase and consumed item. It suppresses further automatic attempts at
that region after an ambiguous click, drop, timeout or unchanged food
until a new manual calibration. No hardcoded food-plate coordinates,
synthetic SOP use messages, arbitrary potions, unproven healing spells or
equipment manipulation are introduced.

`nn status` and `nn food status` include `food_calibrations`/calibration
counts. The existing food model continues saving `food.ckpt`; the
actuator file contains only verified region fingerprints, not new weight
formats. Both are flushed before neural game-over restore, while the
existing rolling native saves `nn-1`, `nn-2`, `nn-3` and five-second
recovery remain unchanged. **This is bounded, manually requested food-plate use after
manual item-to-cursor/equipment setup**, not yet autonomous retrieval from
backpacks or unattended healing.

**Native food-use verification and learned recovery preferences (EOB3 only).**
The game's own food interaction is supported: holding rations in the cursor,
opening the equipment view, and clicking the native food plate consumes the
item and raises the character's food value when hungry (see
`docs/roadmap.md`, Consumables). The food-plate coordinates and potion
subtype semantics are **not** verified in the neural controller. This
iteration therefore adds a *bounded, opt-in instrumented native experiment*,
rather than inventing a click or a healing effect.

With `nn auto on` and `nn train on`, first pick up a documented food
item into the game's native in-hand cursor. While the intended character is
hungry, run `nn food probe <party-slot-0..5>`. The controller records the
actual EOB3 food item class (rations 1347, apple 1451, iron rations 1456),
the cursor item, party identity, level, HP and food counters, and suspends
other autonomous movement/combat while the probe is armed. Use the existing
native food-plate action to consume the item within 30 seconds.
**Only** when the same PC remains present, food actually increases and
the game empties the item cursor is it counted as a verified consumption.
A dropped/equipped item, full stomach, unchanged food, invalid or replaced
party, level transition, unsupported item class or timeout produces
`unverified` and **no training reward**.

The recovery model learns a bounded reward proportional to the observed
food increase, keyed by the known **food item class** and the target
character's food/HP and resource context. Observed HP change is reported
separately and is **not** attributed as potion healing. The existing online
model's checksum-checked v2 format persists this separate food policy as
`Documents/ThirdeyeNN/food.ckpt`, autosaved every eight confirmed
observations, on clean exit, and before rolling-game-save recovery.
Existing `brain.ckpt`, `survival.ckpt`, `combat.ckpt`,
`weaknesses.ckpt` and native `nn-1`.. `nn-3` game saves remain
unchanged. Corrupt existing `food.ckpt` files are preserved, not replaced.

`nn food status` reports the trial's state; `nn status` adds
`food_samples`, `food_reward`, `food_checkpoint` and `food_probe`.
After it has learned confirmed outcomes, `nn food recommend <slot>`
uses the saved recovery policy to choose among food item classes actually
carried by the party. **Recommendations are diagnostic only** until the
native food-plate hit region and item-to-cursor transfer are independently
verified for autonomous execution. No arbitrary potion, healing spell,
equipment coordinate or rest command is issued.

**Survival retreat and resource-aware learning (EOB3 only).**
When autonomy is enabled, the controller reads each party member's native
PC HP/max HP (class 1369 offsets 171/173), food (177), and distinct inventory
references (inventory[0..25] at offset 81). Known ration/apple/iron-ration
classes (1347, 1451, 1456) and potential potion classes (1357..1364)
are counted as **carried resources**. The potion classes are not decoded
into healing subtypes; the controller never assumes all potions heal.

**New executed action:** when a living monster is verified directly ahead,
the weakest member has below 50% HP (or below 65% with no known carried
rations or potions), and a backward/sideways cell is unblocked, neural
autonomy prioritizes a retreat step instead of attacking. Only verified
back/strafe SOP key codes are used; forward and rotation are excluded
from this survival decision. An independent softmax learner uses minimum
HP ratio, average food and resource-availability features. On the next
host action tick it evaluates the actual adjacent step, party identity,
HP/food changes and remaining front-cell monster. Escaping without damage
is rewarded; stalled moves, continued exposure, HP loss and food loss are
penalized. Invalid or reloaded party snapshots do not train. Each decision
and confirmed result logs HP/food deltas, resource counts, reward, mean
reward, samples and checkpoint state. Existing one-second action pacing remains.

Survival experience is saved independently as
Documents/ThirdeyeNN/survival.ckpt with the existing checksum-protected
THIRDEYE_NN 2 model format. It reloads for later decisions, autosaves every
eight confirmed survival outcomes and flushes on shutdown and before
game-over restore. A corrupt existing sidecar is never overwritten.
The control command nn status reports survival_samples, survival_reward,
survival_success and survival_checkpoint. Legacy brain.ckpt, combat.ckpt,
weaknesses.ckpt and the native nn-1/nn-2/nn-3 game saves stay unchanged.

**Limit:** This executes injury-aware defensive movement, **not potion
drinking, healing spells, food consumption or rest**. Those actions need
verified native item-specific effects and use controls. Carried resource
counts do not prove an item is an effective cure.

**Autonomous EOB3 combat (first supported interaction).** When explicit `nn auto on` autonomy
is active, a living enemy immediately ahead on the same floor triggers
right-click weapon attacks through the existing event queue. The agent
observes NPC HP from class 1622 and selects an alive party member (slots
0, 1, or 3) using the verified EOB3 weapon-icon coordinates. A separate
online softmax policy rewards observed damage and penalizes non-damaging
attacks; invalid or unobservable target transitions are not trained on.
The mouse press is released on the next host pump, and menus, manual
mode, and scene resets cancel pending attack feedback. Combat is disabled
in Dungeon Hack until its character offsets and button coordinates have
been verified. It does not auto-consume items or cast healing spells; see the separate survival retreat behaviour above.

The neural movement brain automatically loads from the operating system's
resolved Documents folder, at `Documents/ThirdeyeNN/brain.ckpt`.
Combat learning independently loads/saves
`Documents/ThirdeyeNN/combat.ckpt`. Both use the existing
checksum-protected `THIRDEYE_NN 2` format: no migration or change to
prior movement checkpoints is required. Combat autosaves every 16 learned
outcomes and on shutdown. `nn status` reports `combat_samples`,
`combat_reward`, `combat_success`, and `combat_checkpoint`.
The directory is created as needed. It saves **silently** after 32
new autonomous training results and on clean exit. The checkpoint uses
atomic replacement, a version header, a checksum, and strict numeric
validation. Version 2 saves self-learning outcomes and model weights.
Older version-1 checkpoints remain loadable; their weights are used as
an initial policy, but from then onward **only the agent's own measured
results update the model**. To begin from an untrained policy instead,
back up or remove the old brain file.

When `THIRDEYE_CTL` is enabled on POSIX, the same model is
available to the debug control socket. Commands:
`nn status`, `nn train on/off` (toggle *autonomous
outcome learning*, not player observation), `nn auto on/off`,
`nn predict`, `nn step`, `nn spell status <slot>`,
`nn spell probe <slot> <mage|cleric>`,
`nn food probe <slot>`, `nn food use <slot>`, `nn food status`,
`nn food recommend <slot>`,
`nn save <path>`, and `nn load <path>`.
`nn teach` is disabled; player demonstrations never train
the model. Manual checkpoint commands do not change the automatic
Documents brain path. The live learning loop works without a control
socket on Windows, Linux, and macOS.

Neural unit tests cover outcome scoring, exploration, reward learning,
checkpoint safety, and version-1 migration in
`apps/tests/nn_mode_tests.cpp` and
`apps/tests/nn_checkpoint_tests.cpp`.
