# **Smart AI - Loan Approval System**

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
│── data/               # Dataset used for training & testing
│── models/             # Saved machine learning models
│── scripts/            # Scripts for data preprocessing & training
│── app.py              # Streamlit application
│── requirements.txt    # Python dependencies
│── README.md           # Documentation
```

## **Installation & Setup**
### **Prerequisites**
Ensure you have the following installed on your system:
- Python (>=3.8)
- pip (Python package manager)
- Azure CLI (for deployment management)

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

## **Deployment on Azure**
### **Steps to Deploy on Azure**
1. **Login to Azure:**
   ```bash
   az login
   ```

2. **Create an Azure Web App (if not already created):**
   ```bash
   az webapp create --resource-group <your-resource-group> --plan <your-app-service-plan> --name <your-app-name> --runtime "PYTHON:3.8"
   ```

3. **Deploy the Application:**
   ```bash
   az webapp up --name <your-app-name> --resource-group <your-resource-group>
   ```

4. **Access the Application:**
   - Open: `https://your-app-name.azurewebsites.net`

## **Screenshots**
### **Home Page**
![Home Page](screenshots/home.png)

### **Prediction Result Page**
![Prediction Result](screenshots/result.png)

## **Contributing**
We welcome contributions! Please follow these steps:
1. **Fork the Repository**
2. **Create a Feature Branch** (`git checkout -b feature-name`)
3. **Commit Your Changes** (`git commit -m "Added new feature"`)
4. **Push to Your Fork** (`git push origin feature-name`)
5. **Create a Pull Request**

## **License**
This project is licensed under the **MIT License**.

## **Contact**
For queries or suggestions, contact:
📧 Email: your-email@example.com
📌 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

