import tensorflow as tf
import pandas as pd
from django.shortcuts import render

def indexpage(request):
    data = pd.read_csv("./latest_data.csv")

    print('Shape of the data: ', data.shape)

    data.drop(data.loc[data['emotion']==27].index, inplace=True)
    print(data.shape)
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')

    #Making all letters lowercase
    data['sentence'] = data['sentence'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    #Removing Punctuation, Symbols
    data['sentence'] = data['sentence'].str.replace('[^\w\s]',' ')
    #Removing Stop Words using NLTK
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    data['sentence'] = data['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    #Lemmatisation
    from textblob import Word
    data['sentence'] = data['sentence'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    #Correcting Letter Repetitions
    import re
    def de_repeat(text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)
    #%%
    data['sentence'] = data['sentence'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

    from sklearn import preprocessing
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(data.emotion.values)
    # Splitting into training and testing data in 90:10 ratio
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(data.sentence.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

    import ktrain
    from ktrain import text

    print(data.emotion.value_counts())

    X_train = data.sentence.tolist()
    X_test = data.sentence.tolist()

    y_train = data.emotion.tolist()
    y_test = data.emotion.tolist()

    class_names = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadnes','ssurprise']

    (x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=128, max_features=12800)
    model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), 
                             val_data=(x_test, y_test),
                             batch_size=32)
    learner.fit_onecycle(2e-5, 5)
    learner.validate(val_data=(x_test, y_test), class_names=class_names)
    predictor = ktrain.get_predictor(learner.model, preproc)
    predictor.get_classes()

    import time 

    message = 'I am not happy. It feels like something is wrong'

    start_time = time.time() 
    prediction = predictor.predict(message)

    print('predicted: {} ({:.2f})'.format(prediction, (time.time() - start_time)))

    # let's save the predictor for later use
    predictor.save("test")
    reloaded_model = ktrain.load_predictor('test')
    a1 = reloaded_model.predict('I am happy')
    print(a1)
    # tf.train.write_graph(predictor, "/content/export_dir",
    #                      'saved_model.pb', as_text=False)
    context = {}
    return render(request, 'index.html', context)
