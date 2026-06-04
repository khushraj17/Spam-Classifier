import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn
ps = PorterStemmer()
nltk.download('punkt_tab')
nltk.download('stopwords')



examples = [
    # Spam Messages
    "Congratulations! You have won ₹50,000. Claim your prize now by clicking the link.",
    "URGENT! Your account has been suspended. Verify your details immediately.",
    "You have been selected for a free iPhone 16. Reply YES to claim.",
    "Earn ₹10,000 per day working from home. Limited spots available!",
    "Exclusive offer! Get 90% discount on all products. Shop now.",
    "Winner! You have won a vacation package. Call now to redeem.",
    "Get a personal loan instantly with zero documentation. Apply today.",
    "Free recharge worth ₹999. Click here to activate.",
    "Your ATM card will be blocked. Update KYC immediately.",
    "Claim your lottery winnings before midnight.",
    "You have won a brand new car! Click the link to claim now.",
    "Limited-time offer! Buy one get three free. Order today.",
    "Your Paytm account has been locked. Verify your identity now.",
    "Get rich fast! Earn ₹1 lakh per week from home.",
    "Congratulations! Your mobile number has won a cash prize of ₹5 lakh.",

    "Hey, are we still meeting at 5 PM today?",
    "Don't forget to bring the assignment tomorrow.",
    "Happy Birthday! Have a wonderful day ahead.",
    "Can you call me when you're free?",
    "Your Amazon order has been delivered successfully.",
    "Let's have lunch together after class.",
    "The meeting has been rescheduled to Monday morning.",
    "I have shared the notes in the WhatsApp group.",
    "Mom said dinner will be ready by 8 PM.",
    "Good luck for your exam tomorrow!",
    "Your train will arrive at platform number 3.",
    "Please submit the project report before Friday.",
    "Thank you for your payment. Transaction completed successfully.",
    "The movie starts at 7:30 PM. Don't be late.",
    "I'm outside your house. Come down when you're ready."
]


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectoizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("🛡️ SMS Spam Detector")
st.caption("Powered by Machine Learning")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]


    prob = model.predict_proba(vector_input)[0]

    spam_prob = prob[1] * 100

    
    # 4. Display result
    if result == 1:
        st.error("This is a spam message.")
    else:
        st.success("This is not a spam message.")


    st.progress(int(spam_prob))
    st.write(f"Spam Probability: {spam_prob:.2f}%")


    st.subheader("Try more Messages")

    for i, msg in enumerate(examples, start=1):
        st.markdown(f"**Example {i}**")
        st.code(msg, language="text")


else:


    st.subheader("📋 Preset Messages for Testing")

    for i, msg in enumerate(examples, start=1):
        st.markdown(f"**Example {i}**")
        st.code(msg, language="text")

