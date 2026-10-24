# Support Ticket Classifier

Classifies support tickets into billing, login_access, bug_report, feature_request or account_management using TF-IDF + Logistic Regression.

## Run
    pip install -r requirements.txt
    python classifier.py train
    python classifier.py predict "I was charged twice, please refund"

## Important
tickets.csv is a small hand-written demo set with repeated near-duplicate lines, so the 100% test score is inflated by leakage. Do not quote it. Replace it with a real public dataset (for example a Kaggle customer-support ticket dataset) and report honest metrics.
