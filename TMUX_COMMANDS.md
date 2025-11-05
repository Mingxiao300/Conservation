# Tmux Commands for Running Models

## Python Model - Quick Start

### Option 1: Use the helper script (easiest)

```bash
cd /n/home00/msong300/conservation
./start_python_tmux.sh
```

Then attach:
```bash
tmux attach -t python_model
```

---

### Option 2: Manual tmux commands

**Step 1: Create a new tmux session:**
```bash
tmux new-session -d -s python_model -c /n/home00/msong300/conservation
```

**Step 2: Run the Python model in the session:**
```bash
tmux send-keys -t python_model "srun --mem=16G --time=6:00:00 --pty bash -c 'cd /n/home00/msong300/conservation && ./run_python_model.sh'" Enter
```

**Step 3: Attach to the session:**
```bash
tmux attach-session -t python_model
# Or shorter:
tmux attach -t python_model
```

---

## Attaching to Sessions

### List all tmux sessions:
```bash
tmux list-sessions
# Or shorter:
tmux ls
```

### Attach to Python model session:
```bash
tmux attach-session -t python_model
# Or shorter:
tmux attach -t python_model
```

### Attach to R model session (if you create one):
```bash
tmux attach-session -t r_model
# Or shorter:
tmux attach -t r_model
```

---

## Useful Tmux Commands (while inside a session)

### Detach from session (keeps it running):
- Press: `Ctrl+B`, then press `D`
- Or type: `tmux detach`

### Split window horizontally:
- Press: `Ctrl+B`, then press `"`

### Split window vertically:
- Press: `Ctrl+B`, then press `%`

### Switch between panes:
- Press: `Ctrl+B`, then press arrow keys

### Scroll up/down:
- Press: `Ctrl+B`, then press `[`
- Use arrow keys to scroll
- Press `q` to exit scroll mode

### Kill current pane:
- Press: `Ctrl+B`, then press `x`

### Kill current session:
- Press: `Ctrl+B`, then press `&` (Shift+7)

---

## Creating Both Sessions

### Create Python session:
```bash
cd /n/home00/msong300/conservation
./start_python_tmux.sh
```

### Create R session (separate terminal or tmux window):
```bash
tmux new-session -d -s r_model -c /n/home00/msong300/conservation
tmux send-keys -t r_model "srun --mem=16G --time=6:00:00 --pty bash -c 'cd /n/home00/msong300/conservation && ./install_r_packages_robust.sh'" Enter
```

### Attach to either:
```bash
# Python
tmux attach -t python_model

# R
tmux attach -t r_model
```

---

## Monitoring Progress (from outside tmux)

### View logs:
```bash
# Latest Python log
tail -f /n/home00/msong300/conservation/logs/python_*.log | tail -50

# Latest R log
tail -f /n/home00/msong300/conservation/logs/r_*.log | tail -50
```

### Check if session is running:
```bash
tmux list-sessions | grep python_model
tmux list-sessions | grep r_model
```

### Check outputs:
```bash
ls -lh /n/home00/msong300/conservation/outputs/models/
ls -lh /n/home00/msong300/conservation/outputs/diagnostics/
ls -lh /n/home00/msong300/conservation/outputs/plots/
```

---

## Quick Reference

```bash
# Create Python session
tmux new-session -d -s python_model -c /n/home00/msong300/conservation
tmux send-keys -t python_model "srun --mem=16G --time=6:00:00 --pty bash -c 'cd /n/home00/msong300/conservation && ./run_python_model.sh'" Enter

# Attach to Python session
tmux attach -t python_model

# List sessions
tmux ls

# Detach (while inside): Ctrl+B, then D
```


