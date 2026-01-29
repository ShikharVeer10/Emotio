# 🚀 Quick Reference: Push SentiFlow to GitHub

## One-Line Deploy

```bash
bash CREATE_AND_PUSH_SENTIFLOW.sh
```

Then follow the prompts!

---

## What This Does

1. ✅ Creates `~/SentiFlow` directory
2. ✅ Copies all Python modules from `project-root/` to `src/`
3. ✅ Creates README.md, requirements.txt, .gitignore, LICENSE
4. ✅ Initializes git repository
5. ✅ Creates initial commit
6. ✅ Provides push commands

---

## After Running the Script

### 1. Create GitHub Repository
- Visit: https://github.com/new
- Name: **SentiFlow**
- Don't initialize with anything
- Click "Create repository"

### 2. Push to GitHub
```bash
cd ~/SentiFlow
git remote add origin https://github.com/YOUR_USERNAME/SentiFlow.git
git branch -M main
git push -u origin main
```

### 3. Done! 🎉
Visit: `https://github.com/YOUR_USERNAME/SentiFlow`

---

## Full Documentation

See: **PUSH_SENTIFLOW_INSTRUCTIONS.md** for detailed instructions and troubleshooting.

---

## Can't Run the Script?

Manual steps in **PUSH_SENTIFLOW_INSTRUCTIONS.md**
