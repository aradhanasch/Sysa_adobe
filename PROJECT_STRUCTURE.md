# 📁 Adobe "Connecting the Dots" Challenge - Project Structure

<div align="center">

![Project Structure](https://img.shields.io/badge/Structure-Organized-green?style=for-the-badge)
![Clean Architecture](https://img.shields.io/badge/Architecture-Clean-blue?style=for-the-badge)
![Ready to Deploy](https://img.shields.io/badge/Status-Ready-orange?style=for-the-badge)

**Clean, Organized, and Production-Ready Project Structure**

</div>

---

## 🎯 **Project Overview**

This document outlines the **clean and organized structure** of the Adobe "Connecting the Dots" Challenge solution. All files have been properly organized, nested folders have been eliminated, and all path references have been corrected for seamless operation.

---

## 📂 **Root Directory Structure**

```
hackathon_adobe/
├── 📁 app/                          # Main application directory
│   ├── 📁 input/                    # PDF input files (mounted in Docker)
│   ├── 📁 output/                   # JSON output files (mounted in Docker)
│   ├── 📁 models/                   # ML model storage
│   ├── 📁 round1b/                  # Round 1B persona analysis
│   ├── 📁 tests/                    # Unit tests
│   ├── 📁 __pycache__/              # Python cache files
│   ├── 📄 main.py                   # Main entry point
│   ├── 📄 outline_extractor.py      # Core PDF extraction logic
│   ├── 📄 utils.py                  # Utility functions
│   ├── 📄 train_ml.py               # ML model training
│   ├── 📄 train_heading_classifier.py # Heading classifier training
│   ├── 📄 visualize_headings.py     # Visualization utilities
│   └── 📄 outline_extractor_config.json # Configuration file
├── 📄 README.md                     # Comprehensive project documentation
├── 📄 approach_explanation.md       # Technical approach documentation
├── 📄 final_summary.md              # Project summary and results
├── 📄 requirements.txt              # Python dependencies
├── 📄 Dockerfile                    # Main Docker configuration
├── 📄 Dockerfile.train              # Training Docker configuration
├── 📄 create_training_data.py       # Training data generation
├── 📄 quick_train.py                # Quick ML training script
├── 📄 test_round1b.py               # Round 1B testing
├── 📄 test_volume.py                # Docker volume testing
├── 🚀 run_extraction.bat            # PDF extraction execution
├── 🚀 run_round1b.bat               # Round 1B execution
├── 🚀 run_simple_round1b.bat        # Simple Round 1B execution
├── 🚀 train_ml.bat                  # ML training execution
└── 🚀 train_ml_simple.bat           # Simple ML training execution
```

---

## 🔧 **Key Directories Explained**

### **📁 app/ - Main Application**
The core application directory containing all Python code and resources:

| Directory | Purpose | Contents |
|-----------|---------|----------|
| **input/** | PDF Input | PDF files to be processed |
| **output/** | JSON Output | Generated outlines and analysis |
| **models/** | ML Models | Trained machine learning models |
| **round1b/** | Round 1B Code | Persona-driven analysis modules |
| **tests/** | Unit Tests | Test files for validation |

### **📄 Core Python Files**

| File | Purpose | Key Features |
|------|---------|--------------|
| **main.py** | Entry Point | Processes all PDFs in input directory |
| **outline_extractor.py** | Core Logic | Advanced PDF parsing and ML classification |
| **utils.py** | Utilities | Text normalization and helper functions |
| **train_ml.py** | ML Training | Custom model training pipeline |
| **visualize_headings.py** | Visualization | Debug and analysis tools |

---

## 🚀 **Execution Scripts**

### **Batch Files for Easy Execution**

| Script | Purpose | Command |
|--------|---------|---------|
| **run_extraction.bat** | PDF Outline Extraction | `.\run_extraction.bat` |
| **run_round1b.bat** | Persona Analysis | `.\run_round1b.bat` |
| **run_simple_round1b.bat** | Simple Persona Analysis | `.\run_simple_round1b.bat` |
| **train_ml.bat** | ML Model Training | `.\train_ml.bat` |
| **train_ml_simple.bat** | Simple ML Training | `.\train_ml_simple.bat` |

### **Python Scripts for Development**

| Script | Purpose | Usage |
|--------|---------|-------|
| **create_training_data.py** | Generate Training Data | `python create_training_data.py` |
| **quick_train.py** | Quick ML Training | `python quick_train.py` |
| **test_round1b.py** | Round 1B Testing | `python test_round1b.py` |
| **test_volume.py** | Docker Volume Testing | `python test_volume.py` |

---

## 🐳 **Docker Configuration**

### **Dockerfiles**

| File | Purpose | Target |
|------|---------|--------|
| **Dockerfile** | Main Application | Production deployment |
| **Dockerfile.train** | Training Environment | ML model training |

### **Volume Mounts**
All Docker containers use consistent volume mounts:
- **Input**: `D:/hackathon_adobe/app/input:/app/input`
- **Output**: `D:/hackathon_adobe/app/output:/app/output`

---

## 📊 **File Organization Benefits**

### **✅ Clean Structure**
- **No nested folders**: Eliminated confusing nested `hackathon_adobe/hackathon_adobe/` structure
- **Logical grouping**: Related files are organized in appropriate directories
- **Clear separation**: Application code, documentation, and scripts are clearly separated

### **✅ Path Consistency**
- **All batch files updated**: Corrected path references in all execution scripts
- **Docker paths standardized**: Consistent `/app/input` and `/app/output` mounting
- **Python imports working**: All module imports use correct relative paths

### **✅ Production Ready**
- **Docker optimized**: Clean container structure for deployment
- **Script automation**: One-click execution for all operations
- **Error handling**: Proper error handling and logging throughout

---

## 🔍 **Path References Fixed**

### **Updated Files**
All path references have been corrected in the following files:

| File | Old Path | New Path |
|------|----------|----------|
| **run_extraction.bat** | `hackathon_adobe/hackathon_adobe/app/` | `hackathon_adobe/app/` |
| **run_round1b.bat** | `hackathon_adobe/hackathon_adobe/app/` | `hackathon_adobe/app/` |
| **run_simple_round1b.bat** | `hackathon_adobe/hackathon_adobe/app/` | `hackathon_adobe/app/` |
| **train_ml.bat** | `hackathon_adobe/hackathon_adobe/app/` | `hackathon_adobe/app/` |
| **train_ml_simple.bat** | `hackathon_adobe/hackathon_adobe/app/` | `hackathon_adobe/app/` |

### **Docker Paths**
All Docker containers use the correct absolute paths:
- **Input Volume**: `/app/input`
- **Output Volume**: `/app/output`
- **Models Directory**: `/app/models`

---

## 🎯 **Usage Instructions**

### **1. Quick Start**
```bash
# Build Docker image
docker build -t pdf-intelligence:latest .

# Run PDF extraction
.\run_extraction.bat

# Run persona analysis
.\run_round1b.bat
```

### **2. Development Mode**
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python app/tests/test_outline_extractor.py

# Train ML model
python app/train_ml.py
```

### **3. Docker Deployment**
```bash
# Production deployment
docker run --rm \
  -v $(pwd)/app/input:/app/input \
  -v $(pwd)/app/output:/app/output \
  pdf-intelligence:latest
```

---

## 📈 **Quality Assurance**

### **✅ Structure Validation**
- **No nested folders**: Clean, flat structure
- **Consistent naming**: Standardized file and directory names
- **Proper organization**: Logical grouping of related files

### **✅ Path Validation**
- **All batch files updated**: Correct path references
- **Docker paths verified**: Consistent volume mounting
- **Python imports tested**: All modules import correctly

### **✅ Functionality Testing**
- **PDF processing**: Works with current structure
- **ML training**: Model training pipeline functional
- **Persona analysis**: Round 1B analysis operational

---

## 🏆 **Project Status**

### **✅ Ready for Competition**
- **Clean architecture**: Professional, organized structure
- **Production ready**: Dockerized deployment
- **Comprehensive docs**: Complete documentation
- **Error free**: All path issues resolved

### **✅ Ready for Development**
- **Modular design**: Easy to extend and modify
- **Clear separation**: Distinct components and responsibilities
- **Testing framework**: Unit tests included
- **Debug tools**: Comprehensive logging and visualization

---

<div align="center">

**🎉 Project Structure Successfully Reorganized!**  
**🚀 Ready for Adobe "Connecting the Dots" Challenge!**

</div> 