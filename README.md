## Speaker's Coverage Prediction Application

### Overview
Welcome to the Speaker's Coverage Prediction Application! This is a fun and educational project where I've been experimenting with Python and machine learning to predict the coverage area of speaker systems. Whether you're a tech enthusiast, a student, or just curious about machine learning, this tool is designed to provide a straightforward glimpse into how data can be used to model real-world phenomena.

### Features
- **Learning by Doing**: Dive into the code to see how Python handles data with libraries like NumPy and pandas.
- **First Steps in Machine Learning**: See the K-Nearest Neighbors (KNN) algorithm in action. Don't worry if you're new to it; it's all part of the learning curve!
- **Interactive Input**: Try inputting different values for speaker dimensions and power. It's a hands-on way to see how inputs affect outputs.
- **Data Visualization**: (Coming Soon) I hope to add some plots and charts to make it easier to understand the relationships in the data.

### Installation

If you're new to Python or programming in general, here’s how you can get started:

1. **Clone the repository**:
   ```
   git clone https://github.com/VuongMinhKhanh/Speaker-s-Coverage-Evaluation-KNN.git
   ```
2. **Install required libraries**:
   Make sure Python is installed on your system first. Then run this command in your terminal to install the libraries:
   ```
   pip install numpy pandas matplotlib scikit-learn
   ```

### Usage

Run the script from your command line like so:

```
python Speaker's_Coverage.py
```

You'll need to enter some details about your speaker. Don't worry, the script will guide you through it:
- Length, width, and height of the speaker
- Power of the speaker
- Parameters for the KNN model (`n` and `p`)

The program will then tell you the predicted coverage area based on your inputs.

### Functions Description

- **find_space_keyword**: A function to play around with text searching.
- **output_excel**: Learn how data can be saved in Excel format.
- **remove_str and cal_means_power**: See how strings can be converted and manipulated into numbers.
- **testKNN**: Get a peek into model testing with different settings.
- **user_input**: This is where we get our data from the user.

### Contributing

If you’re also learning and want to suggest improvements or add new features, feel free to fork the repository and explore. Your contributions are encouraged!

### Acknowledgments

- Big thanks to all the open-source tutorials and documentation that helped me understand these concepts.
- Appreciation to anyone who’s taken the time to review or improve this project.

### Future Enhancements

- **Data Visualization**: I'm planning to add some visual elements to help us see what's happening behind the scenes.
- **User-Friendly Interface**: Thinking about making a simple web interface for this tool someday.
