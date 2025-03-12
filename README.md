# Smart AI - Loan Approval System

## **Overview**
The **Smart AI - Loan Approval System** is an AI-powered fintech application that predicts loan approvals with high accuracy. The system employs a **Random Forest Classifier (93% accuracy)** and provides **AI-driven responses** for loan decisions, ensuring transparency. The project is built using **Streamlit** and is **deployed on Azure** for scalability.

## **Key Features**
- 🏦 **AI-Powered Loan Approval Prediction** using **Random Forest Classifier**.
- 🔍 **High Accuracy (93%)** in loan eligibility determination.
- 🤖 **AI-Driven Explanations** for approval decisions, enhancing transparency.
- 🚀 **Deployed on Azure**, ensuring scalability and high availability.
- 📊 **Interactive UI with Streamlit** for seamless user experience.

## **Technologies Used**
- **Machine Learning**: Random Forest Classifier
- **Data Handling**: SMOTE (for handling imbalanced datasets)
- **Deployment**: Microsoft Azure
- **Frontend**: Streamlit

## **Project Structure**
```bash
Smart-AI-Loan-Approval/
│── loan_approval_dataset.csv
│── preprocessed_data.csv       # Dataset used for training & testing
│── credit_scoring_model.pkl    # Saved machine learning models
│── preprocessing.py            # Scripts for data preprocessing
│── model.py                    # Scripts for model training
│── Smart_loan_approval.py                      # Streamlit application
│── requirements.txt            # Python dependencies
│── README.md                   # Documentation
```

## **Installation & Setup**
### **Prerequisites**
Ensure you have the following installed on your system:
- Python (3.11)
- pip (Python package manager)

### **Steps to Run the Project Locally**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/Smart-AI-Loan-Approval.git
   cd Smart-AI-Loan-Approval
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the Web Application:**
   - Open your browser and visit: `http://localhost:8501/`


## **Screenshots**
### **Home Page**
![Home Page](screenshots/home.png)

### **Prediction Result Page**
![Prediction Result](screenshots/result.png)

## **License**
This project is licensed under the **MIT License**.

## **Contact**
For queries or suggestions, contact:
📧 Email: jagdeeshpersonal@gmail.com
📌 LinkedIn: [Jagdeesh P](https://www.linkedin.com/in/jagdeesh-p/)

