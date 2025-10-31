# OpenSearch JVector Testing Scripts

This directory contains scripts for testing OpenSearch JVector functionality, particularly with large indices.

## Project Structure

```
scripts/
├── jvector_index_and_search/          # JVector indexing and search testing
│   ├── README.md                      # Comprehensive documentation
│   ├── create_and_test_large_index.py # Main testing script
├── demo.sh                            # Demo script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Quick Links

- **[JVector Index and Search Testing](jvector_index_and_search/README.md)** - Main testing framework
- **[Testing Guide](jvector_index_and_search/TESTING_RECALL.md)** - How to test recall measurement
- **[Package Documentation](jvector_index_and_search/jvector_utils/README.md)** - Utilities API reference

## Installation

### Prerequisites

- Python 3.6+
```bash
#Linux users
sudo apt install python3

# Mac users (install latest version of python3)
sudo brew install python3
```
- OpenSearch instance with JVector plugin installed. 
   See Install Jvector plugin section of [developer guide] (DEVELOPER_GUIDE.md).

### Setup

#### Using a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

1. Create a virtual environment:
   ```bash
   # Linux users
   sudo apt install python3.11-venv   

   # Using venv (Python 3.3+)
   python3 -m venv .venv
   
   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. Install the required Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. When you're done, you can deactivate the virtual environment:
   ```bash
   deactivate
   ```

#### Direct Installation

Make sure you have pip installed. If you prefer not to use a virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

### JVector Index and Search Testing

The main testing framework is located in the `jvector_index_and_search/` directory.

**See the [jvector_index_and_search/README.md](jvector_index_and_search/README.md) for complete documentation.**

#### Quick Start

```bash
cd jvector_index_and_search

# Run unit tests (no OpenSearch required)
python test_recall_measurement.py

# Run integration tests (requires OpenSearch)
python test_recall_integration.py

# Create and test a large index
python create_and_test_large_index.py --num-vectors 100000

# With recall measurement
python create_and_test_large_index.py --measure-recall --num-vectors 100000
```

#### Key Features

- **Large-scale indexing**: Index millions of vectors with configurable batch sizes
- **Force merge tracking**: Monitor graph merge and quantization times
- **Search testing**: Perform multiple searches with detailed JVector statistics
- **Recall measurement**: Memory-efficient ground truth tracking (~1000x less memory)
- **Performance visualization**: Generate plots of merge times vs document count

#### Common Commands

```bash
cd jvector_index_and_search

# Basic large index test
python create_and_test_large_index.py --num-vectors 1000000

# With recall measurement
python create_and_test_large_index.py --measure-recall --num-vectors 100000 --num-recall-queries 20

# Performance analysis with visualization
python create_and_test_large_index.py --force-merge-frequency 100000 --csv-output merge_times.csv --plot

# Search only (skip indexing)
python create_and_test_large_index.py --skip-indexing --index my-existing-index --dimension 768
```

For complete documentation, options, and examples, see:
- **[jvector_index_and_search/README.md](jvector_index_and_search/README.md)** - Complete usage guide
- **[jvector_index_and_search/TESTING_RECALL.md](jvector_index_and_search/TESTING_RECALL.md)** - Testing and troubleshooting

## Other Scripts

### demo.sh

A demo script for quick testing (if available).

## Profiling

You can profile the OpenSearch Java process while running tests:

```bash
# Get the process id of the opensearch java process
PID=$(jps | grep OpenSearch | awk '{print $1}')

# Start profiling
jcmd $PID JFR.start name=OnDemand settings=profile duration=600s filename=/tmp/app_jfr_$(date +%s).jfr
```

## Contributing

When adding new scripts or modifying existing ones:
1. Follow the modular structure established in `jvector_index_and_search/`
2. Add comprehensive documentation
3. Include unit tests where applicable
4. Update this README with links to new functionality