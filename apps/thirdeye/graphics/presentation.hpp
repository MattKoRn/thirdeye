#pragma once

#include <SDL3/SDL.h>

namespace GRAPHICS {

// SDL3 fullscreen uses the current desktop display mode by default, not
// exclusive resolution switching. Combine it with BORDERLESS to make the
// intended window style explicit. RESIZABLE is useful for the --windowed
// escape hatch (and F11's restored window).
constexpr SDL_WindowFlags gameWindowFlags(bool windowed) {
    return windowed ? SDL_WINDOW_RESIZABLE
                    : (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS |
                       SDL_WINDOW_RESIZABLE);
}

// The game always renders to a fixed 320x200 canvas. STRETCH fills the entire
// target window, including widescreen displays, with NO letterbox/black bars.
// SDL's logical event conversion then maps mouse clicks back to 320x200.
constexpr SDL_RendererLogicalPresentation GAME_PRESENTATION =
    SDL_LOGICAL_PRESENTATION_STRETCH;

} // namespace GRAPHICS
