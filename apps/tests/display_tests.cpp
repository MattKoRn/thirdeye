#include <gtest/gtest.h>

#include "../thirdeye/graphics/presentation.hpp"

TEST(DisplayPresentation, FullscreenUsesDesktopWithoutBorders) {
    const SDL_WindowFlags flags = GRAPHICS::gameWindowFlags(false);
    EXPECT_NE(flags & SDL_WINDOW_FULLSCREEN, 0u);
    EXPECT_NE(flags & SDL_WINDOW_BORDERLESS, 0u);
    EXPECT_NE(flags & SDL_WINDOW_RESIZABLE, 0u);
}

TEST(DisplayPresentation, WindowedModeKeepsNormalResizableFrame) {
    const SDL_WindowFlags flags = GRAPHICS::gameWindowFlags(true);
    EXPECT_EQ(flags & SDL_WINDOW_FULLSCREEN, 0u);
    EXPECT_EQ(flags & SDL_WINDOW_BORDERLESS, 0u);
    EXPECT_NE(flags & SDL_WINDOW_RESIZABLE, 0u);
}

TEST(DisplayPresentation, CanvasFillsScreenWithoutLetterboxOrCrop) {
    EXPECT_EQ(GRAPHICS::GAME_PRESENTATION, SDL_LOGICAL_PRESENTATION_STRETCH);
    EXPECT_NE(GRAPHICS::GAME_PRESENTATION, SDL_LOGICAL_PRESENTATION_LETTERBOX);
    EXPECT_NE(GRAPHICS::GAME_PRESENTATION, SDL_LOGICAL_PRESENTATION_OVERSCAN);
}
