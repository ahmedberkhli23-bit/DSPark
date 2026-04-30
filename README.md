# 🎧 DSPark - Build advanced audio tools with ease

[![Download DSPark](https://img.shields.io/badge/Download-DSPark-blue.svg)](https://github.com/ahmedberkhli23-bit/DSPark)

## 🎯 About this software

DSPark provides a collection of tools for digital audio processing. It handles complex tasks like filtering, reverb, and signal analysis. Creators use it to build audio plugins and applications.

The framework functions as a header-only library. This means you do not need to compile external libraries or link complicated files. You include one header file in your code, and the engine works immediately. It relies on C++20 for speed and stability.

This software prioritizes real-time performance. It uses SIMD instructions to process audio signals without lag or stuttering. Whether you need a compressor, an equalizer, or a reverb effect, this library supplies the components you need to get the job done.

## 🛠️ System requirements

To use DSPark, ensure your computer meets these standards:

- Operating System: Windows 10 or Windows 11.
- Processor: Any modern 64-bit CPU.
- Storage: At least 50 megabytes of free space for your project files.
- Development Software: A C++ compiler that supports the C++20 standard. Visual Studio 2022 or newer is recommended.
- Memory: 8 gigabytes of RAM or more.

## 📥 How to download and set up

Follow these steps to obtain the files for your Windows machine:

1. Visit [the official download page](https://github.com/ahmedberkhli23-bit/DSPark) to access the latest version.
2. Look for the button labeled "Code" on the main repository page.
3. Select "Download ZIP" from the menu.
4. Save the folder to a location on your computer where you keep your development projects.
5. Right-click the folder and choose "Extract All" to unzip the files.
6. Open your preferred code editor or Integrated Development Environment (IDE).

## ⚙️ Integrating the library

DSPark stays true to its header-only design. Integration requires minimal effort.

1. Locate the include directory within the extracted folder.
2. Copy the path to this directory.
3. In your project settings within your IDE, navigate to the "Additional Include Directories" section.
4. Paste the path you copied into this field.
5. Add the following line to the top of your C++ source file: `#include "DSPark.hpp"`.

Once you perform these steps, your project gains access to all DSPark functions. You can now instantiate filters, compressors, and audio objects directly in your code.

## 🚀 Creating your first effect

The library organizes tools into logical groups. To create a simple audio filter, follow this pattern:

1. Declare the object in your main loop or processor class.
2. Initialize the parameters, such as cutoff frequency or gain, using the provided setter functions.
3. Pass your audio buffer into the process function of the object.

The library handles the underlying math. You provide the input signal, and the code returns the modified audio samples instantly.

## 🎛️ Available features

DSPark includes a wide array of tools for professional audio work:

- Equalizers: Sculpt frequency responses with precision.
- Compressors: Manage the dynamic range of your signals.
- FFT Analysis: Transform time-based signals into frequency data for visualization or processing.
- Reverb Modules: Add depth and space to sounds using high-performance algorithms.
- SIMD Acceleration: Benefit from hardware-level optimization that ensures low latency.
- Real-time Safety: Use these tools in professional audio threads without risk of dropouts or clicks.

## 📦 File structure

Understanding the file layout helps you navigate the library:

- /include: Contains the main header files.
- /examples: Provides working snippets showing how to implement filters and dynamics.
- /tests: Includes verification scripts to ensure the math functions perform as intended.
- /docs: Stores extra information on specific algorithms if you need a deeper look at the theory.

## ❓ Troubleshooting common issues

If you encounter errors, check these common points:

- Compiler compatibility: DSPark requires a C++20 compliant compiler. Ensure your project settings in Visual Studio are set to "ISO C++20 Standard (/std:c++20)".
- Include paths: If the compiler reports that it cannot find the file, verify that the folder path in your IDE points exactly to the directory containing the header file.
- Memory alignment: When using SIMD features, ensure your data buffers use proper memory alignment to prevent crashes.

## 💡 Best practices for developers

For the best results, adhere to these simple rules:

- Keep process blocks small: Audio engines run on tight schedules. Perform only necessary calculations within your processing loop.
- Use the examples: The provided example files serve as structural templates. Base your features on these patterns.
- Monitor your performance: Use profilers to check the CPU usage of your signal chain. DSPark offers efficient algorithms, but stacking too many complex effects can eventually affect performance.
- Update regularly: Check the link below periodically for improvements and new additions to the framework.

[Download the latest release here](https://github.com/ahmedberkhli23-bit/DSPark)