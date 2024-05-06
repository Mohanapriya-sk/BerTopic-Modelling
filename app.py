from flask import Flask, render_template
import pandas as pd
from bertopic import BERTopic
from preprocess import preprocess_text

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    # Load your YouTube comments data from CSV or other formats
    df = pd.read_excel('data/comments_data_latest.xlsx')
    df['comment_text'] = df['comment_text'].astype(str)
    df['preprocessed_text'] = df['comment_text'].apply(preprocess_text)
    return df

def perform_topic_modeling(df):
    # Validate DataFrame contents
    if df.empty or 'comment_text' not in df.columns:
        print("Error: DataFrame is empty or missing 'comment_text' column")
        return df, None  # Return empty model if DataFrame is invalid

    # Preprocess text and prepare for topic modeling
    df['preprocessed_text'] = df['comment_text'].apply(preprocess_text)

    # Initialize BERTopic model
    model = BERTopic()

    # Fit BERTopic model
    topics, _ = model.fit_transform(df['preprocessed_text'])

    # Assign topics to DataFrame
    df['topic'] = topics

    # Get topic info from BERTopic model
    topic_info = model.get_topic_info()

    # Print topic_info to inspect its structure
    print("Topic Info:")
    print(topic_info.head())  # Print first few rows of topic_info DataFrame

    # Retrieve and assign topic labels to comments
    if not topic_info.empty:
        # Create a dictionary mapping topic IDs to representative words (keywords)
        topic_labels = {row['Topic']: row['Representation'] for _, row in topic_info.iterrows()}
        df['topic_label'] = df['topic'].map(topic_labels).fillna('Unknown')
        # Get topic frequencies and sort by frequency (Count)
        topic_frequent_words = topic_info.sort_values('Count', ascending=False)
    else:
        df['topic_label'] = 'Unknown'  # Assign default label if no topic info available
        topic_frequent_words = pd.DataFrame(columns=['Topic', 'Count', 'Name', 'Representation', 'Representative_Docs'])

    return df, model, topic_frequent_words


@app.route('/')
def index():
    # Load and preprocess data
    df = load_and_preprocess_data()

    # Perform BERTopic modeling and retrieve topic information
    df, model, topic_frequent_words = perform_topic_modeling(df)

    return render_template('index.html', topic_frequent_words=topic_frequent_words, df=df)


if __name__ == '__main__':
    app.run(debug=True)
