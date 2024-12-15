from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample email dataset (Text and Labels)
emails = [
    "Congratulations, you have won a lottery! Click here to claim your prize.",
    "Urgent! Your account is at risk. Update your password immediately.",
    "Don't miss our sale. Huge discounts on all products!",
    "Meeting scheduled for tomorrow at 3 PM.",
    "Team lunch this Friday. Let us know your preferences.",
    "Invoice for your recent purchase attached.",
    "Your subscription is about to expire. Renew now to avoid interruptions.",
    "Looking forward to catching up next week!",
    "Reminder: Submit your project report by Monday.",
    "Win a brand-new car! Participate in our survey today!"
]

labels = ["Spam", "Spam", "Spam", "Not Spam", "Not Spam", "Not Spam", "Spam", "Not Spam", "Not Spam", "Spam"]

# Step 1: Convert text data into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)  # Convert email text to feature vectors
y = labels

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = classifier.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with a new email
new_email = ["Congratulations! You've been selected for a free gift. Reply to claim now."]
new_email_vectorized = vectorizer.transform(new_email)
new_prediction = classifier.predict(new_email_vectorized)
print("\nNew Email Prediction:", new_prediction[0])
