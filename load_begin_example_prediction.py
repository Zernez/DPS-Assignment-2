
#Loading model and vocab
print("Loading pre-trained model")
vocabulary_to_load = pickle.load(open("vocab.pickle", 'rb'))
count_vect = CountVectorizer(vocabulary=vocabulary_to_load)
load_model = pickle.load(open("model.pickle", 'rb'))count_vect._validate_vocabulary()
tfidf_transformer = tf_idf(categories)[0]


#predicting the streaming kafka messages
consumer = KafkaConsumer('twitter-stream',bootstrap_servers=\['localhost:9092'])
print("Starting ML predictions.")
for message in consumer:
X_new_counts = count_vect.transform([message.value])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = load_model.predict(X_new_tfidf)
print(message.value+" => "+fetch_train_dataset(categories).target_names[predicted[0]])