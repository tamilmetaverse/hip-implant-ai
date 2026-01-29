# How to Record a Demo Video/GIF

This guide shows you how to create a professional demo video or GIF of the Hip Implant AI system.

## Quick Demo

For a visual, animated demonstration:

```bash
python demo_visual.py
```

This will run through all the steps with nice formatting and timing.

---

## Option 1: Screen Recording (Recommended)

### Using Windows Built-in Screen Recorder

1. **Open the demo directory in terminal**
   ```bash
   cd c:\Users\tamil\hip_implant_ai
   ```

2. **Start recording**
   - Press `Win + G` to open Xbox Game Bar
   - Click the Record button (or press `Win + Alt + R`)

3. **Run the visual demo**
   ```bash
   python demo_visual.py
   ```

4. **Stop recording**
   - Press `Win + Alt + R` again
   - Video saved to: `C:\Users\tamil\Videos\Captures\`

5. **Convert to GIF** (optional)
   - Use [ezgif.com](https://ezgif.com/video-to-gif) (free online tool)
   - Upload your MP4, trim it, and download as GIF

---

## Option 2: ScreenToGif (Best for GIFs)

### Download and Install

1. Download from: https://www.screentogif.com/
2. Install and open ScreenToGif
3. Click "Recorder"
4. Position the recording window over your terminal
5. Click "Record"
6. Run: `python demo_visual.py`
7. Click "Stop" when done
8. Edit and save as GIF

**Pros:**
- Free and open-source
- Direct GIF creation
- Built-in editor
- Small file sizes

---

## Option 3: Terminal Recording (Advanced)

### Using asciinema (Terminal Recorder)

```bash
# Install
pip install asciinema

# Record
asciinema rec demo.cast

# Run demo
python demo_visual.py

# Stop with Ctrl+D

# Convert to GIF
# Use https://github.com/asciinema/agg or similar
```

---

## Option 4: Static Output (No Recording Needed)

Create a static output file that looks good on GitHub:

```bash
python demo_visual.py > demo_output.txt
```

Then create a "Demo Output" section in README with:

````markdown
## Demo Output

```
======================================================================
                    HIP IMPLANT AI - DEMONSTRATION
======================================================================

[Step 1] Checking Environment
  [OK] Python 3.13.2 detected
  [OK] PyTorch 2.10.0+cpu installed
  [OK] All dependencies verified
...
```
````

---

## Tips for Great Recordings

### Terminal Settings

1. **Increase font size**
   - Right-click terminal title bar
   - Properties → Font → Size 16 or 18

2. **Use a clean theme**
   - Consider using Windows Terminal with a nice color scheme
   - Or use PowerShell with custom colors

3. **Full screen terminal**
   - Press `F11` or maximize window
   - Hide unnecessary UI elements

### Recording Settings

- **Resolution**: 1920x1080 (Full HD) or 1280x720 (HD)
- **Frame rate**: 30 FPS is enough for terminal
- **Duration**: Keep it under 1 minute for GIFs
- **File size**: Aim for <10MB for GIFs (GitHub limit is 10MB)

---

## Adding to GitHub README

### Option A: Embed GIF

```markdown
## Demo

![Hip Implant AI Demo](demo.gif)
```

### Option B: Link to Video

```markdown
## Demo

[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
```

### Option C: Collapsible Demo Output

```markdown
## Demo

<details>
<summary>Click to see demo output</summary>

\`\`\`
[Demo output here]
\`\`\`

</details>
```

---

## Recommended Workflow

**For GitHub (Best):**
1. Use ScreenToGif to record terminal
2. Trim to essential parts (30-60 seconds)
3. Optimize to <5MB
4. Save as `demo.gif`
5. Add to repository: `git add demo.gif`
6. Update README with: `![Demo](demo.gif)`

**For Presentations:**
1. Use Windows Game Bar
2. Record full demo
3. Export as MP4
4. Use in PowerPoint/presentations

**For Quick Sharing:**
1. Run `python demo_visual.py > output.txt`
2. Share the text output
3. Works everywhere, no file size limits

---

## Need Help?

- ScreenToGif: https://www.screentogif.com/
- ezgif converter: https://ezgif.com/
- asciinema: https://asciinema.org/
- GitHub GIF guide: https://github.blog/2021-05-06-github-gif-upload-support/

---

**Ready to record?** Run `python demo_visual.py` and create your demo!
