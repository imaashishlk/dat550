# DAT550 Project
Project as on DAT550 - Data Mining and Deep Learning

**_We used Google Collab in deriving the final file. The initial cells used in linking the Google Drive and visualizing the directory can be ignored._**

The project aims at exploring the various machine learning techniques for detection and analysis of fake news from real news. We are comparing several classifying techniques and implementing neural network approach to explore the model and to detect false news.

**Packages Used:**
- Numpy
- Pandas
- Seaborn
- Regex
- NLTK
- SKLearn 
- TensorFlow
- Keras

Every packages has been imported to the notebook file and the system has to have these installed. The packages which are uncommon are listed in the notebook which are installed in the running time.

For example, 

- **!pip install tensorflow_hub**
- **!pip install tensorflow_text**

The notebook file can be easily run to reproduce the results as shown on the report. The pre-run outputs are still demonstrated on the notebook file.

**Note:** To run LSTM Model you need to install glove.6B.100d

We can install glove.6B from https://nlp.stanford.edu/projects/glove/

**Dataset Used:**

Under the data folder (submission):
- Politifact (politifact.json)
- Snopes (snopes.json)

**DATASET LINK**: https://drive.google.com/file/d/1OcSQW1_bqahgKn5krQWY9Le8VGyY2utX/view

1. **Data Collection**: Import the necessary libraries. We have two dataset saved as json files, the first file contains data from politifact website and the second file contains data from snopes website. The two dataset consist of multiclassification label.

- Politifact data contains the following columns: 'claim', 'doc', 'label', 'factchecker', 'published', 'speaker', 'date_stated', 'stated_in', 'url', 'topic', 'sources', 'summary'

- Snopes data contains the following columns: 'label', 'claim', 'doc', 'factchecker', 'published', 'url', 'topic', 'sources', 'extra_description'

2. **Exploring and Data visualization:**

- Printing column name of each data sets and choosing most common columns like label,claim and doc.
- Merging two datasets into one data and finding the unique label using thresold function.
- Replacing different label names with the multilabels of true or false.
- Both 'claim' and 'doc' column contains text, so merging those columns as 'text' field for better way
- Representing the binary values for label field '0 for false' and '1 for true'

3. **Preprocessing & Cleaning Dataset:** The dataset has to be preprocessed so that the machine learning algorithm can detect patterns easily. By using the 'nltk' libraries to removing the following functionalities to get exact information from the text.

- Removal of Punctuation Marks and Special Characters
- Removal of Stopwords.
- Lemmatization

4. **Splitting the data into training and testing parts**

5. **Feature extraction & Classification Models**

- Using TF-IDF bagofwords & pre-trainded word Embedding GloVe.
- Applying simple Machine learning Model and check the accuracy.
- Applying Deep Neural Network Classifier (LSTM, Bert) and make display all visualization using tensorboard.

6. **Evaluating the models and comparing between classifier models.**

**Contributors:**

- Shaima Ahmad Freja
- Vinothini Aravindan
- Aashish Karki
- Yeganeh Hallaj
