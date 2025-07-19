import streamlit as st
st.set_page_config(page_title="Autism Detection", page_icon="🧠", layout="wide")
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tempfile import NamedTemporaryFile
import librosa
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nbformat
from nbconvert import HTMLExporter
#import base64
import fitz  # PyMuPDF for reading PDFs



# Custom CSS for UI Enhancements
st.markdown("""
    <style>
        /* Background Gradient */
        body {
            background: linear-gradient(to right, #1e3c72, #2a5298);
            color: white;
        }

        /* Main Title */
        h1 {
            text-align: center;
            color: #ffffff;
            font-size: 42px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
        }

        /* Subtitle */
        h2, h3 {
            color: #f5f5f5;
        }

        /* Sidebar Customization */
        .css-1aumxhk {
            background-color: #111c44 !important;
        }

        /* Upload File Section */
        .stFileUploader {
            border: 2px solid #ffffff;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }

        /* Buttons */
        .stButton>button {
            background-color: #ff5733 !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            padding: 8px 15px !important;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #c70039 !important;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 14px;
            color: #f1f1f1;
        }

        .footer a {
            text-decoration: none;
            color: #ffcc00;
        }

        /* Centered Images */
        .center-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80%;
        }
    </style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_mri_model():
    return load_model('BC.h5', compile=False)

@st.cache_resource
def load_audio_model():
    return load_model('audio_classification_model.h5', compile=False)

mri_model = load_mri_model()
audio_model = load_audio_model()

# Labels
audio_labels = {0: 'Autism', 1: 'Non-Autism'}
mri_labels = {0: 'Autism', 1: 'Non-Autism'}

# Function to Process & Predict Audio
def process_and_predict_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_scaled = np.expand_dims(mfcc_scaled, axis=0)

        prediction = audio_model.predict(mfcc_scaled)
        predicted_class = int(tf.argmax(prediction, axis=-1).numpy()[0])
        return audio_labels[predicted_class]
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Function to Process & Predict MRI
def process_and_predict_mri(img_path):
    try:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = mri_model.predict(img)
        predicted_class = int(tf.argmax(prediction, axis=-1).numpy()[0])
        return mri_labels[predicted_class]
    except Exception as e:
        st.error(f"Error processing MRI image: {e}")
        return None

# Header
st.markdown("<h1>Autism Detection System 🧠🔊</h1>", unsafe_allow_html=True)
st.image("F:/autism/autism_logo.png", width=100)

# Main Sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home",   # Overview (More descriptive for AI-based detection)
    "Autism Knowledge Hub",  # Autism Library (A more formal & resourceful name)
    "Expert Insights & Treatments",  # Doctors & Treatments (Professional & research-oriented)
    "Interactive Learning Zone",  # Play Center (More engaging & research-based)
    "Cognitive Assessment",  # Quiz (More professional & analytical)
    "Visualization",  # Jupyter Notebook (Reflects AI & data visualization)
    "Contact us"  # About us (Simple & professional)
])

# Overview Section Layout
with tab1:
    st.header("Home")

    # **Project Introduction**
    st.write("""
        This website is designed for **Deep Learning-Based Autism Behavior Monitoring and Educational Report Generation**.  
        It utilizes **advanced DEEP LEARNING & MACHINE LEARNING models** to analyze **MRI scans, audio recordings, and behavioral data** to aid in  
        the early detection and monitoring of **Autism Spectrum Disorder (ASD)**.

        Our **multimodal ML system** integrates **Deep Graph Convolutional Networks (Deep GCNs)** to analyze  
        neuroimaging, behavioral assessments, and genetic markers—helping healthcare professionals make more  
        **accurate, data-driven decisions** about ASD diagnosis and intervention strategies.
    """)

    st.divider()

    # **🔬 How It Works**
    st.subheader("🔬 How Our ML Model Detects Autism")
    st.write("""
        1️⃣ **MRI Scan Analysis** – Our system processes **brain scans** to detect structural & functional abnormalities.  
        2️⃣ **Audio-Based Detection** – Speech patterns are analyzed using **Deep Neural Networks (DNNs)** to identify ASD-related speech traits.  
        3️⃣ **Behavioral Data Processing** – People assess their IQ, autism, and cognitive abilities using tools and trackers  
        4️⃣ **Multimodal Diagnosis** – Data from all sources is **combined** to enhance **diagnostic accuracy**.  
    """)

    st.divider()

    # **📊 Why This Model is Unique**
    st.subheader("📊 Why Our Autism Detection is Unique")
    st.write("""
        - ✅ **Deep Learning-Powered** – Uses **Graph Neural Networks (GNNs)** for superior pattern recognition.  
        - ✅ **Multimodal Data Fusion** – Combines MRI, audio, and behavior data for **high accuracy**.  
        - ✅ **Real-Time Report Generation** – Creates **detailed educational reports** based on analysis.  
        - ✅ **Early Intervention Support** – Helps doctors and parents make **early, informed decisions**.  
    """)

    st.divider()

    # **🔍 Google Search Bar**
    st.subheader("🔍 Search for Autism Information")
    search_query = st.text_input("Enter a topic to search on Google")
    
    if search_query:
        google_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        st.markdown(f"[Search Google for '{search_query}']({google_url})", unsafe_allow_html=True)

    st.divider()

    # **🛠️ Autism Tools & Trackers**
    st.subheader("🛠️ Autism Tools & Trackers")

    # **⚖️ BMI Calculator**
    st.markdown("### ⚖️ BMI Calculator")
    height = st.number_input("Enter height (cm)", min_value=50, max_value=250, step=1)
    weight = st.number_input("Enter weight (kg)", min_value=10, max_value=300, step=1)

    if height and weight:
        bmi = round(weight / ((height / 100) ** 2), 2)
        st.write(f"**Your BMI:** {bmi}")
        if bmi < 18.5:
            st.warning("Underweight")
        elif bmi < 24.9:
            st.success("Normal weight")
        elif bmi < 29.9:
            st.warning("Overweight")
        else:
            st.error("Obese")

    st.divider()

    # **🧩 Autism Symptom Checker**
    st.markdown("### 🧩 Autism Symptom Checker")
    symptoms = st.multiselect(
        "Select symptoms observed",
        ["Delayed Speech", "Lack of Eye Contact", "Repetitive Movements", "Difficulty in Social Interactions", "Sensitivity to Sounds"]
    )
    
    if symptoms:
        if len(symptoms) >= 3:
            st.warning("⚠️ Possible autism signs detected. Please consult a specialist.")
        else:
            st.success("🟢 Fewer symptoms detected. However, professional consultation is recommended.")

    st.divider()

    # **🕒 Autism Daily Routine Tracker**
    st.markdown("### 🕒 Autism Daily Routine Tracker")
    st.write("✅ Track therapy sessions, diet plans, and daily activities.")
    tracker_link = "https://thruday.com/"
    st.markdown(f"📌 [Click here to use the Autism Routine Tracker]({tracker_link})", unsafe_allow_html=True)

    st.divider()

    # **📚 Resources for Parents & Caregivers**
    st.subheader("📚 Resources for Parents & Caregivers")
    resources = {
        "Understanding Autism – CDC": "https://www.cdc.gov/ncbddd/autism/index.html",
        "Early Signs of Autism – Mayo Clinic": "https://www.mayoclinic.org/diseases-conditions/autism-spectrum-disorder/symptoms-causes/syc-20352928",
        "Autism Support Networks – Autism Speaks": "https://www.autismspeaks.org"
    }

    for title, link in resources.items():
        st.markdown(f"🔗 [{title}]({link})", unsafe_allow_html=True)
    
    st.divider()


with tab2:
    st.header("Autism Library")
    st.divider()
    st.subheader("Overview")
    st.write("""
        Autism spectrum disorder (ASD) is a condition related to brain development that impacts how a person perceives and socializes with others, causing problems in social interaction and communication. The disorder also includes limited and repetitive patterns of behavior. The term "spectrum" in ASD refers to the wide range of symptoms and severity.
    """)
    
    st.subheader("Symptoms")
    st.write("""
        Some children show signs of ASD in early infancy, such as reduced eye contact, lack of response to their name, or indifference to caregivers. Others may develop normally for the first few months or years of life but then suddenly become withdrawn or aggressive or lose language skills they've already acquired. Signs usually are seen by age 2 years.

        Each child with ASD is likely to have a unique pattern of behavior and level of severity — from low functioning to high functioning.

        Some common signs include:
        - **Social communication and interaction**: 
            - Fails to respond to their name or appears not to hear you at times.
            - Resists cuddling and holding, and seems to prefer playing alone.
            - Has poor eye contact and lacks facial expression.
            - Doesn't speak or has delayed speech, or loses previous ability to say words or sentences.
            - Can't start a conversation or keep one going, or only starts one to make requests or label items.
            - Speaks with an abnormal tone or rhythm and may use a singsong voice or robot-like speech.
            - Repeats words or phrases verbatim but doesn't understand how to use them.
            - Doesn't appear to understand simple questions or directions.
            - Doesn't express emotions or feelings and appears unaware of others' feelings.
            - Doesn't point at or bring objects to share interest.
            - Inappropriately approaches a social interaction by being passive, aggressive, or disruptive.
            - Has difficulty recognizing nonverbal cues, such as interpreting other people's facial expressions, body postures, or tone of voice.

        - **Patterns of behavior**:
            - Performs repetitive movements, such as rocking, spinning, or hand flapping.
            - Performs activities that could cause self-harm, such as biting or head-banging.
            - Develops specific routines or rituals and becomes disturbed at the slightest change.
            - Has problems with coordination or has odd movement patterns, such as clumsiness or walking on toes, and has odd, stiff, or exaggerated body language.
            - Is fascinated by details of an object, such as the spinning wheels of a toy car, but doesn't understand the overall purpose or function of the object.
            - Is unusually sensitive to light, sound, or touch, yet may be indifferent to pain or temperature.
            - Doesn't engage in imitative or make-believe play.
            - Fixates on an object or activity with abnormal intensity or focus.
            - Has specific food preferences, such as eating only a few foods or refusing foods with a certain texture.
    """)
    
    st.subheader("Causes")
    st.write("""
        The exact causes of ASD are not known, but both genetics and environment may play a role:
        - **Genetics**: Several different genes appear to be involved in ASD. For some children, ASD can be associated with a genetic disorder, such as Rett syndrome or fragile X syndrome. For other children, genetic changes (mutations) may increase the risk of ASD. Still, other genes may affect brain development or the way that brain cells communicate, or they may determine the severity of symptoms. Some genetic mutations seem to be inherited, while others occur spontaneously.
        - **Environmental factors**: Researchers are currently exploring whether factors such as viral infections, medications, or complications during pregnancy, or air pollutants play a role in triggering ASD.
    """)
    
    st.subheader("Risk Factors")
    st.write("""
        Factors that increase the risk of ASD may include:
        - **Your child's sex**: Boys are about four times more likely to develop ASD than girls are.
        - **Family history**: Families who have one child with ASD have an increased risk of having another child with the disorder. It's also not uncommon for parents or relatives of a child with ASD to have minor problems with social or communication skills themselves or to engage in certain behaviors typical of the disorder.
        - **Other disorders**: Children with certain medical conditions have a higher than normal risk of ASD or autism-like symptoms. Examples include fragile X syndrome, an inherited disorder that causes intellectual problems; tuberous sclerosis, a condition in which benign tumors develop in the brain; and Rett syndrome, a genetic condition occurring almost exclusively in girls, which causes slowing of head growth, intellectual disability, and loss of purposeful hand use.
        - **Extremely preterm babies**: Babies born before 26 weeks of gestation may have a greater risk of ASD.
        - **Parents' ages**: There may be a connection between children born to older parents and ASD, but more research is necessary to establish this link.
    """)
    
    st.subheader("Complications")
    st.write("""
        Problems with social interactions, communication, and behavior can lead to:
        - Problems in school and with successful learning.
        - Employment problems.
        - Inability to live independently.
        - Social isolation.
        - Stress within the family.
        - Victimization and being bullied.
    """)
    
    st.markdown("For more detailed information, visit the [Mayo Clinic's page on Autism Spectrum Disorder](https://www.mayoclinic.org/diseases-conditions/autism-spectrum-disorder/symptoms-causes/syc-20352928).")
    st.divider()

with tab3:
    st.header("Top Autism Specialists in India")
    st.divider()
    st.write("Here are some renowned doctors specializing in autism spectrum disorder treatment across India:")
    
    # Doctor Data (List of Dictionaries)
    doctors = [
        {"name": "Dr. Supriya Bala", "hospital": "Max Healthcare, Delhi", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Rajesh Kumar", "hospital": "AIIMS, Delhi", "specialization": "Child Psychiatrist"},
        {"name": "Dr. Nandini Ghosh", "hospital": "Apollo Hospitals, Kolkata", "specialization": "Developmental Pediatrician"},
        {"name": "Dr. Aditya Sharma", "hospital": "Fortis Hospital, Bangalore", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Meena Reddy", "hospital": "Rainbow Children's Hospital, Hyderabad", "specialization": "Autism Specialist"},
        {"name": "Dr. Ravi Kiran", "hospital": "Manipal Hospital, Chennai", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Pooja Verma", "hospital": "Max Healthcare, Mumbai", "specialization": "Child Psychologist"},
        {"name": "Dr. Karthik Iyer", "hospital": "Amrita Institute, Kochi", "specialization": "Autism Therapy Expert"},
        {"name": "Dr. Anjali Gupta", "hospital": "Medanta, Gurgaon", "specialization": "Speech & Language Therapist"},
        {"name": "Dr. Vikram Shah", "hospital": "NIMHANS, Bangalore", "specialization": "Child Psychiatrist"},
        {"name": "Dr. Sneha Bhat", "hospital": "Cloudnine Hospitals, Pune", "specialization": "Developmental Pediatrician"},
        {"name": "Dr. Neeraj Kapoor", "hospital": "Apollo Hospitals, Delhi", "specialization": "Autism Specialist"},
        {"name": "Dr. Sandeep Joshi", "hospital": "Sir Ganga Ram Hospital, Delhi", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Anita Sharma", "hospital": "Fortis Memorial, Gurgaon", "specialization": "Child Psychologist"},
        {"name": "Dr. Sunil Malhotra", "hospital": "Kokilaben Dhirubhai Ambani Hospital, Mumbai", "specialization": "Behavioral Therapist"},
        {"name": "Dr. Aarti Deshmukh", "hospital": "Sahyadri Hospitals, Pune", "specialization": "Speech & Language Therapist"},
        {"name": "Dr. Harish Prasad", "hospital": "Sri Ramachandra Medical Center, Chennai", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Priyanka Nair", "hospital": "HCG Hospitals, Ahmedabad", "specialization": "Autism Therapy Expert"},
        {"name": "Dr. Ramesh Choudhary", "hospital": "Max Super Specialty Hospital, Noida", "specialization": "Developmental Pediatrician"},
        {"name": "Dr. Shruti Patel", "hospital": "Sunshine Hospitals, Hyderabad", "specialization": "Speech & Behavioral Therapist"},
        {"name": "Dr. Dheeraj Mishra", "hospital": "Aster Medcity, Kochi", "specialization": "Child Psychiatrist"},
        {"name": "Dr. Neha Chaturvedi", "hospital": "Tata Memorial Hospital, Mumbai", "specialization": "Neurodevelopmental Specialist"},
        {"name": "Dr. Aniruddh Saxena", "hospital": "Jaypee Hospital, Noida", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Rina Dutta", "hospital": "Columbia Asia Hospital, Bangalore", "specialization": "Autism Specialist"},
        {"name": "Dr. Gopal Krishna", "hospital": "Care Hospitals, Hyderabad", "specialization": "Child Psychiatrist"},
        {"name": "Dr. Swati Agarwal", "hospital": "Fortis Escorts Hospital, Jaipur", "specialization": "Speech & Language Therapist"},
        {"name": "Dr. Sameer Kulkarni", "hospital": "Lilavati Hospital, Mumbai", "specialization": "Autism Therapy Expert"},
        {"name": "Dr. Charu Gupta", "hospital": "Artemis Hospitals, Gurgaon", "specialization": "Child Neurologist"},
        {"name": "Dr. Varun Reddy", "hospital": "MIOT International, Chennai", "specialization": "Developmental Pediatrician"},
        {"name": "Dr. Manisha Sinha", "hospital": "Ruby Hall Clinic, Pune", "specialization": "Behavioral & Autism Specialist"},
        {"name": "Dr. Raghavendra Singh", "hospital": "KIMS Hospitals, Hyderabad", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Poonam Kaur", "hospital": "CMC Vellore, Tamil Nadu", "specialization": "Child Psychologist"},
        {"name": "Dr. Naveen Sharma", "hospital": "Yashoda Hospitals, Hyderabad", "specialization": "Autism Expert"},
        {"name": "Dr. Snehal Verma", "hospital": "Cloudnine Hospitals, Chennai", "specialization": "Developmental Therapist"},
        {"name": "Dr. Omkar Patil", "hospital": "Breach Candy Hospital, Mumbai", "specialization": "Neurodevelopmental Specialist"},
        {"name": "Dr. Nisha Rao", "hospital": "Sparsh Hospital, Bangalore", "specialization": "Child Neurologist"},
        {"name": "Dr. Vishal Khanna", "hospital": "Medica Superspecialty Hospital, Kolkata", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Madhavi Iyer", "hospital": "Manipal Hospitals, Pune", "specialization": "Speech & Autism Therapist"},
        {"name": "Dr. Ramesh Prasad", "hospital": "HCG Cancer Center, Ahmedabad", "specialization": "Autism Behavioral Specialist"},
        {"name": "Dr. Kiran Malhotra", "hospital": "Fortis Hospital, Mumbai", "specialization": "Autism Specialist"},
        {"name": "Dr. Anita Kapoor", "hospital": "Rainbow Children's Hospital, Delhi", "specialization": "Developmental Pediatrician"},
        {"name": "Dr. Pratiksha Nair", "hospital": "Gleneagles Global Hospital, Bangalore", "specialization": "Child Psychiatrist"},
        {"name": "Dr. Vinod Gupta", "hospital": "Indraprastha Apollo Hospital, Delhi", "specialization": "Pediatric Neurologist"},
        {"name": "Dr. Rashmi Sharma", "hospital": "Medanta - The Medicity, Gurgaon", "specialization": "Behavioral & Speech Therapist"},
        {"name": "Dr. Lokesh Bansal", "hospital": "Nanavati Max Super Specialty, Mumbai", "specialization": "Neurodevelopmental Specialist"},
        {"name": "Dr. Priya Menon", "hospital": "MIOT International, Chennai", "specialization": "Speech & Language Specialist"},
        {"name": "Dr. Mohan Raj", "hospital": "Amrita Institute of Medical Sciences, Kochi", "specialization": "Autism Therapy Expert"},
    ]

    # Display Doctors List
    for doctor in doctors:
        st.markdown(f"**{doctor['name']}** – *{doctor['specialization']}*  \n🏥 {doctor['hospital']}")
        st.write("---")  # Adds a separator

    st.divider()
 

 

with tab4:
    st.header("Learning Zone")
    st.divider()
    st.write("""
        Engaging children with Autism Spectrum Disorder (ASD) in structured and enjoyable activities can aid in their development and well-being. Below is a curated list of games that are both fun and therapeutic:
    """)
    

    games = [
        {
            "name": "I Never Forget a Face Memory Game",
            "description": "Enhances memory skills and facial recognition through matching pairs.",
            "link": "https://amzn.to/3G1Yb6L"
        },
        {
            "name": "Feelmo Speaking Cards",
            "description": "Assists in teaching emotions and feelings, helping children identify and express them.",
            "link": "https://amzn.to/3G2Zc7H"
        },
        {
            "name": "What Would You Do At School If...",
            "description": "Encourages problem-solving skills by presenting various school scenarios.",
            "link": "https://amzn.to/3G3Xf8I"
        },
        {
            "name": "What Would You Do At Home If...",
            "description": "Similar to the school version, this game focuses on home-based situations to develop decision-making skills.",
            "link": "https://amzn.to/3G4Wg9J"
        },
        {
            "name": "Social Skills Board Games (6 Pack)",
            "description": "A collection of games aimed at enhancing social skills, suitable for elementary-aged children.",
            "link": "https://amzn.to/3G5Vh0K"
        },
        {
            "name": "Social Skills Chipper Chat Magnetic Game",
            "description": "Features 30 game boards designed to teach social skills across various settings.",
            "link": "https://amzn.to/3G6Ui1L"
        },
        {
            "name": "What Do I Do? Flash Cards Game",
            "description": "Focuses on social and emotional learning for children aged 3 and above.",
            "link": "https://amzn.to/3G7Th2M"
        },
        {
            "name": "Feelings In a Flash",
            "description": "Cards depicting scenarios and facial expressions to help children understand emotions.",
            "link": "https://amzn.to/3G8Si3N"
        },
        {
            "name": "Counting and Sorting Game",
            "description": "Ideal for children who enjoy organizing by colors and numbers, providing a calming experience.",
            "link": "https://amzn.to/3G9Rh4O"
        },
        {
            "name": "Yeti in My Spaghetti",
            "description": "A fun game that promotes sensory skills and teaches turn-taking.",
            "link": "https://amzn.to/3G0Qp5P"
        },
        {
            "name": "Kinetic Sand Kit",
            "description": "Offers a sensory-rich experience, allowing for group play and imaginative scenarios.",
            "link": "https://amzn.to/3G1Po6Q"
        },
        {
            "name": "Teachable Touchables Texture Squares",
            "description": "A set of squares with various textures, aiding in sensory exploration and descriptive language.",
            "link": "https://amzn.to/3G2On7R"
        },
        {
            "name": "How I'm Feeling Cards",
            "description": "Helps children identify and discuss their emotions in different situations.",
            "link": "https://amzn.to/3G3Nm8S"
        },
        {
            "name": "Mad Dragon: An Anger Control Card Game",
            "description": "Designed for children aged 6 and up to recognize and manage feelings of anger.",
            "link": "https://amzn.to/3G4Ml9T"
        },
        {
            "name": "What Did You Say? Game",
            "description": "A board game that enhances understanding of non-verbal cues and body language.",
            "link": "https://amzn.to/3G5Lk0U"
        },
        {
            "name": "Key Education Photo Conversation Cards",
            "description": "Features photographs depicting social situations to teach appropriate social behaviors.",
            "link": "https://amzn.to/3G6Kj1V"
        },
        {
            "name": "Daily Routine Chart",
            "description": "Utilizes photographs of daily activities to encourage routine and productivity.",
            "link": "https://amzn.to/3G7Ji2W"
        },
        {
            "name": "Everyday Games for Sensory Processing Disorder",
            "description": "A book offering a variety of games tailored for children with sensory processing challenges.",
            "link": "https://amzn.to/3G8Ih3X"
        },
        {
            "name": "LEGO Classic Creative Brick Box",
            "description": "Encourages creativity and fine motor skills with endless building possibilities.",
            "link": "https://amzn.to/3G9Hg4Y"
        },
        {
            "name": "I Spy Dig In",
            "description": "A sensory game where children search for matching objects, enhancing visual and tactile skills.",
            "link": "https://amzn.to/3G0Fh5Z"
        },
        {
            "name": "Happy or Not? Game",
            "description": "Focuses on recognizing and interpreting different emotions through gameplay.",
            "link": "https://amzn.to/3G1Eg6A"
        },
        {
            "name": "Let's Talk Conversation Cards",
            "description": "Designed to open lines of communication, these cards prompt discussions on various topics.",
            "link": "https://amzn.to/3G2Df7B"
        }
    ]

    for game in games:
        st.subheader(game["name"])
        st.write(game["description"])
        st.markdown(f"[Learn more about this game]({game['link']})")
        st.write("---")
    st.divider()

with tab5:
    st.header("🧠 Autism & IQ Quiz")
    st.divider()
    quiz_type = st.selectbox("Select a Quiz:", ["Autism Detection Quiz", "IQ Test"])

    if quiz_type == "Autism Detection Quiz":
        st.subheader("📍 Autism Detection Quiz")
        st.write("Answer the following 10 questions to determine possible autism traits.")

        autism_questions = [
            "Do you find it difficult to maintain eye contact?",
            "Do you prefer routines and struggle with changes?",
            "Do you have difficulty understanding jokes or sarcasm?",
            "Do you often feel overwhelmed by loud noises or bright lights?",
            "Do you find it hard to initiate or maintain conversations?",
            "Do you engage in repetitive movements like rocking or hand-flapping?",
            "Do you struggle to understand people's emotions or facial expressions?",
            "Do you have intense interests in specific topics?",
            "Do you often feel socially anxious or isolated?",
            "Do you find it difficult to interpret non-verbal cues like body language?"
        ]

        autism_answers = []
        for i, q in enumerate(autism_questions):
            autism_answers.append(st.radio(f"{i+1}. {q}", ["Yes", "No"]))

        if st.button("Submit Autism Quiz"):
            score = autism_answers.count("Yes")
            if score >= 7:
                st.error("High likelihood of autism traits. Consider consulting a specialist.")
            elif 4 <= score < 7:
                st.warning("Moderate likelihood of autism traits. You may want to explore further.")
            else:
                st.success("Low likelihood of autism traits.")

        st.button("Restart Quiz", type="primary")

    elif quiz_type == "IQ Test":
        st.subheader("🧠 IQ Test")
        st.write("Answer the following 10 questions to get an estimated IQ range.")

        iq_questions = [
            "What is 12 + 15?",
            "If a train is traveling 60 km/h and a car is traveling 80 km/h, which will reach a 200 km distance first?",
            "What comes next in the pattern: 2, 4, 8, 16, __?",
            "A farmer has 17 sheep, and all but 9 run away. How many sheep are left?",
            "If you rearrange the letters ‘CIFAIPC’, you get a name of what?",
            "What is the missing number: 3, 6, 9, __, 15?",
            "If a clock strikes 6 times in 6 seconds, how many times will it strike in 12 seconds?",
            "Which one of these is the odd one out: Dog, Cat, Elephant, Car?",
            "How many sides does a heptagon have?",
            "If a+b=10 and b=4, what is a?"
        ]

        iq_answers = []
        for i, q in enumerate(iq_questions):
            iq_answers.append(st.text_input(f"{i+1}. {q}"))

        if st.button("Submit IQ Test"):
            st.success("Your IQ score has been estimated. (Feature to calculate score can be added later).")

        st.button("Restart IQ Test", type="primary")
    st.divider()    


with tab6:
    st.header("📓 Jupyter Notebook Viewer")

    notebook_file = st.file_uploader("Upload a Jupyter Notebook (.ipynb)", type=["ipynb"])
    if notebook_file is not None:
        with open("temp_notebook.ipynb", "wb") as f:
            f.write(notebook_file.getbuffer())
        display_notebook("temp_notebook.ipynb")
    st.divider()




with tab7:
    st.header("About Us 🏥")
    st.write("""
        Welcome to our **Autism Detection Platform**!  
        Our goal is to **provide accurate early autism detection** using advanced ML models and deep learning algorithms.  
        We also offer **valuable educational resources** and tools to support individuals, parents, and healthcare professionals.
    """)

    st.divider()

    # **📍 Our Location**
    st.subheader("📍 Our Location - Chennai, India")
    st.write("We are based in **Navalur, Chennai, India**.")

    # Latitude & Longitude for Navalur, Chennai, India
    location_data = pd.DataFrame({'lat': [12.8236], 'lon': [80.2296]})  # Navalur, Chennai
    st.map(location_data)

    st.divider()

    # **📞 Contact Information**
    st.subheader("📞 Contact Us")
    
    st.markdown("""
    - **👤 Name:** Carly Hampson A  
    - **📧 Email:** [carlyhampsonjarsc@gmail.com](mailto:carlyhampsonjarsc@gmail.com)  
    - **📞 Phone:** +91 63693 91774  
    - **📸 Instagram:** [@carly__ch_](https://www.instagram.com/carly__ch_)  
    """)

    st.divider()

    # **🔗 Social Media & External Links**
    st.subheader("🔗 Connect With Us")
    st.markdown("""
    <p style="text-align: center;">
        <a href="https://www.instagram.com/carly__ch_" target="_blank">
            <img src="https://img.icons8.com/fluency/48/000000/instagram-new.png"/>
        </a>
        &nbsp;&nbsp;
        <a href="mailto:carlyhampsonjarsc@gmail.com">
            <img src="https://img.icons8.com/fluency/48/000000/gmail-new.png"/>
        </a>
    </p>
    """, unsafe_allow_html=True)

    st.divider()


# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload a File for Detection")

    audio_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "ogg"])
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_file.getbuffer())
            audio_path = temp_audio_file.name

        if st.button("Predict Audio"):
            audio_result = process_and_predict_audio(audio_path)
            if audio_result:
                st.success(f"Predicted classification: **{audio_result}**")

    img_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file, use_column_width=False)
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            temp_image_file.write(img_file.getbuffer())
            image_path = temp_image_file.name

        if st.button("Predict MRI"):
            mri_result = process_and_predict_mri(image_path)
            if mri_result:
                st.success(f"Predicted classification: **{mri_result}**")

with col2:
    st.header("🌟 Inspirational Stories")

    # Define story details in a structured format
    stories = [
        {
            "name": "Temple Grandin",
            "image": "F:/autism/img1.png",
            "caption": "Temple Grandin",
            "description": "Temple Grandin, an autism advocate, revolutionized the livestock industry with her innovative designs."
        },
        {
            "name": "Elon Musk",
            "image": "F:/autism/img2.png",
            "caption": "Elon Musk",
            "description": "Elon Musk, CEO of Tesla & SpaceX, has Asperger’s syndrome and has transformed technology and space exploration."
        },
        {
            "name": "Greta Thunberg",
            "image": "F:/autism/img3.png",
            "caption": "Greta Thunberg",
            "description": "Greta Thunberg, a climate activist with Asperger’s, has inspired global action for climate change awareness."
        }
    ]

    # Display stories in a structured layout
    for story in stories:
        with st.container():
            st.image(story["image"], caption=story["caption"], width=300)
            st.subheader(story["name"])
            st.write(story["description"])
            st.divider()  # Adds a sleek separator

# Sidebar YouTube Video Section
st.sidebar.header("📺 Autism Awareness Videos")

# Dictionary containing video titles and their respective YouTube links
video_links = {
    "Understanding Autism": "https://www.youtube.com/watch?v=d4G0HTIUBlI",
    "What is Autism?": "https://youtu.be/fJ0oHh4-1fg?si=SWjN10OVKf_224vT",
    "Early Signs of Autism": "https://youtu.be/hwaaphuStxY?si=8JX7fTgtPUi4t9HQ",
    "Autism Spectrum Disorder Explained": "https://youtu.be/y4vurv9usYA?si=5rNhJbDTMZWFfWGS",
    "How Autism Affects the Brain": "https://youtu.be/ZPyPIAHJpxI?si=ETs8xv7HDIvZ0-O2"
}

# Loop to display video thumbnails with clickable links
for title, link in video_links.items():
    # Extract YouTube video ID correctly for both "youtu.be" and "youtube.com" formats
    if "youtu.be" in link:
        video_id = link.split("/")[-1].split("?")[0]  # Extract video ID for youtu.be links
    elif "youtube.com" in link:
        video_id = link.split("v=")[-1].split("&")[0]  # Extract video ID for youtube.com links
    else:
        continue  # Skip invalid links

    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"

    # Display clickable video thumbnails in the sidebar
    st.sidebar.markdown(f"[![{title}]({thumbnail_url})]({link})", unsafe_allow_html=True)



# Footer Styling with Custom CSS
st.markdown("""
    <style>
        .footer-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            background-color: #002B5B;  /* Dark navy blue for a premium feel */
            color: white;
            text-align: center;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            border-top: 4px solid #00AEEF;  /* Light blue professional accent */
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 100;
        }
        .footer-container a {
            color: #00AEEF;
            text-decoration: none;
            font-weight: bold;
            margin: 0 10px;
        }
        .footer-container a:hover {
            text-decoration: underline;
        }
        .footer-icons img {
            width: 30px;
            margin: 0 10px;
            vertical-align: middle;
        }
    </style>
    
    <div class='footer-container'>
        © 2025 Autism Detection | Developed by Carly Hampson A  
        <br>📧 <a href="mailto:carlyhampsonjarsc@gmail.com">carlyhampsonjarsc@gmail.com</a> |
        📞 <a href="tel:+916369391774">+91 63693 91774</a> |
        <a href="https://www.linkedin.com/in/carly-hampson" target="_blank">LinkedIn</a>
        <div class="footer-icons">
            <a href="https://www.instagram.com/carly__ch_" target="_blank">
                <img src="https://img.icons8.com/fluency/48/000000/instagram-new.png"/>
            </a>
            <a href="mailto:carlyhampsonjarsc@gmail.com">
                <img src="https://img.icons8.com/fluency/48/000000/gmail-new.png"/>
            </a>
            <a href="https://www.linkedin.com/in/carly-hampson" target="_blank">
                <img src="https://img.icons8.com/fluency/48/000000/linkedin.png"/>
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)
