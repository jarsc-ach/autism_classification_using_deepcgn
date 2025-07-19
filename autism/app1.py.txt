import streamlit as st

# Function to load the autism overview from file
def load_overview():
    with open("F:/autism/autism_overview.txt", "r", encoding="utf-8") as file:
        return file.read()

# Set Streamlit page configuration
st.set_page_config(page_title="Autism Detection", page_icon="🧠", layout="wide")

# Website Title
st.title("Autism Detection Website")

# Sidebar Navigation
menu = ["Overview", "Autism Library", "Symptoms", "Treatments", "About"]
choice = st.sidebar.radio("Navigation", menu)

if choice == "Overview":
    st.header("Overview of Autism Spectrum Disorder")
    st.write(load_overview())  # Load and display text from file

elif choice == "Autism Library":
    st.header("Autism Library")
    st.write("Information about autism cases, research, and news.")

elif choice == "Symptoms":
    st.header("Symptoms of Autism")
    st.write("Details about symptoms, cases, and relevant links.")

elif choice == "Treatments":
    st.header("Autism Treatments")
    st.write("Information on treatment approaches and therapies.")

elif choice == "About":
    st.header("About This Website")
    st.write("Created by Carly Hampson.")
    
    # Social Media Links
    st.markdown("📷 **Instagram:** [carly__ch__](https://instagram.com/carly__ch__)")  
    st.markdown("📧 **Email:** [carlyhampsonjarsc@gmail.com](mailto:carlyhampsonjarsc@gmail.com)")

# Sidebar: YouTube Videos Section
st.sidebar.header("📺 Learn More About Autism")
videos = {
    "What is Autism?": "https://www.youtube.com/watch?v=RbwRrVw-CRo",
    "Signs of Autism": "https://www.youtube.com/watch?v=8O7ZQq9dh_s",
    "Understanding Autism": "https://www.youtube.com/watch?v=8TI4NoKP-OE",
}
for title, url in videos.items():
    st.sidebar.markdown(f"[▶ {title}]({url})")

# Footer
st.markdown("---")
st.markdown("© 2025 Carly Hampson | All Rights Reserved")
