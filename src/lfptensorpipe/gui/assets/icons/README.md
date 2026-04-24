# App Icon Assets

Place application icon assets in this directory.

## Required files
1. Source master icon (raster PNG)
- `app_icon.png`

2. macOS icon
- `macos/lfptensorpipe.icns`

3. Windows icon
- `windows/lfptensorpipe.ico`

4. PNG icon set (for Qt runtime fallbacks and documentation)
- `png/icon_16.png`
- `png/icon_24.png`
- `png/icon_32.png`
- `png/icon_48.png`
- `png/icon_64.png`
- `png/icon_128.png`
- `png/icon_256.png`
- `png/icon_512.png`
- `png/icon_1024.png`

## Notes
- Keep a single visual design across all formats.
- `app_icon.png` is the only supported icon-generation master for this
  directory. Generated PNG sizes, `.ico`, and `.icns` should be rebuilt from
  that file.
- Windows and generic runtime PNG rebuilds must preserve aspect ratio and pad
  the result onto a centered transparent square canvas instead of stretching
  the artwork.
- macOS `.icns` rebuilds must also use the transparent source artwork directly;
  do not add a generated opaque backplate.
- Generated macOS icon assets must remain square and unmasked; do not pre-cut
  rounded corners into generated PNG assets because macOS applies the final
  app-icon presentation itself.
- Recommended primary drawing size: at least 1024x1024.
- Keep safe padding around the logo mark to avoid clipping in dock/taskbar.
