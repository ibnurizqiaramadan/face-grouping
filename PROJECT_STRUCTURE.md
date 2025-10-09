# Project Structure

Face Clustering with Dual GPU - Well-organized and production-ready.

## 📁 Directory Layout

```
face-grouping/
│
├── 🐍 src/                          # Python Source Code
│   ├── app.py                       # Single GPU implementation
│   ├── app_dual_gpu.py              # Dual GPU implementation (main)
│   └── check_redis.py               # Redis data viewer utility
│
├── 📜 scripts/                      # Shell Scripts
│   ├── run_dual_gpu.sh              # Main launcher (dual GPU)
│   ├── run_gpu.sh                   # Single GPU launcher
│   ├── monitor_gpu.sh               # GPU monitoring utility
│   └── README.md                    # Scripts documentation
│
├── 📚 docs/                         # Documentation
│   ├── README.md                    # Documentation index
│   ├── ARCHITECTURE.md              # System architecture
│   ├── CHANGELOG.md                 # Version history
│   ├── CONFIG_GUIDE.md              # Configuration guide
│   ├── FINAL_STATUS.md              # Project status
│   ├── PERFORMANCE.md               # Benchmarks
│   ├── SUMMARY.md                   # Quick overview
│   └── TROUBLESHOOTING.md           # Problem solving
│
├── 📄 Configuration Files
│   ├── README.md                    # Main documentation
│   ├── requirements.txt             # Python dependencies
│   └── .gitignore                   # Git ignore rules
│
└── 📁 Output
    └── clusters_output_labeled/     # Clustering results
        ├── person_0/
        ├── person_1/
        └── ...
```

## 🎯 Quick Navigation

### Getting Started
→ Start here: [README.md](README.md)

### Running the Application
→ Scripts: [scripts/](scripts/) or [scripts/README.md](scripts/README.md)

### Configuration & Optimization
→ Config guide: [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)

### Troubleshooting
→ Problem solving: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

### Understanding the Code
→ Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Performance Tuning
→ Benchmarks: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)

## 📊 File Count Summary

| Category | Count | Location |
|----------|-------|----------|
| Python files | 3 | `src/` |
| Shell scripts | 3 | `scripts/` |
| Documentation | 9 | `docs/` + root |
| Config files | 2 | root |

**Total:** 17 files (excluding output)

## 🔑 Key Files

### Most Important Files

1. **[README.md](README.md)** - Start here
2. **[scripts/run_dual_gpu.sh](scripts/run_dual_gpu.sh)** - Main launcher
3. **[src/app_dual_gpu.py](src/app_dual_gpu.py)** - Main application
4. **[docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)** - Configuration
5. **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Problem solving

### For Developers

- **Source code**: [src/](src/)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Changelog**: [docs/CHANGELOG.md](docs/CHANGELOG.md)

### For Users

- **Quick start**: [README.md](README.md)
- **Configuration**: [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)
- **Help**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 🏗️ Design Principles

1. **Separation of Concerns**
   - Code in `src/`
   - Scripts in `scripts/`
   - Docs in `docs/`

2. **Self-Documenting**
   - README in each folder
   - Clear naming conventions
   - Comprehensive documentation

3. **Production Ready**
   - Error handling
   - Monitoring tools
   - Configuration guides
   - Troubleshooting docs

4. **Scalable**
   - Modular structure
   - Easy to extend
   - Well-organized

---

**Version**: 3.1  
**Last Updated**: 2025-10-09  
**Status**: ✅ Production Ready
