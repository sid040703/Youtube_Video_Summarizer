import yt_dlp
import os
import shutil
from moviepy.editor import VideoFileClip
import openai
import assemblyai as aai
import textwrap
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fpdf import FPDF
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_generator = pipeline("text2text-generation", model="t5-small")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
openai.api_key = "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"
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
app = FastAPI()
api_key = os.getenv("API_KEY")



class YouTubeRequest(BaseModel):
    url: str

@app.post("/process_youtube")
async def process_youtube(request: YouTubeRequest):
    try:
        url = request.url
        
        # Specify the downloads folder for the user (can be modified as per OS)
        summary = extract_summary_from_video(url)
        questions, answers = extract_questions_and_answers(url)
        return {"summary": summary, "questions": questions, "answers": answers}
    
    except Exception as e:
        return {"error": str(e)}


def clear_download_folder(folder_path):
    """
    This function removes all files and subdirectories from the given folder.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory and its contents
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")

# Example Usage
def download_youtube_video(url, output_path):
    ydl_opts = {
        "format": "mp4",
        "outtmpl": f"{output_path}/%(title)s.%(ext)s"
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def find_first_mp4(folder_path):
    """
    Finds the first .mp4 file in the given folder and returns its full path.
    
    :param folder_path: The directory to search for .mp4 files.
    :return: Full path of the first .mp4 file found, or None if not found.
    """
    try:
        files = os.listdir(folder_path)  # Get all files in the directory
        for file in files:
            if file.lower().endswith('.mp4'):  # Case insensitive check
                return os.path.join(folder_path, file)
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied for folder '{folder_path}'.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None  # No .mp4 file found or error occurred

def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as an audio file.
    
    :param video_path: Path to the input video file.
    :param audio_path: Path to save the extracted audio file.
    """
    try:
        with VideoFileClip(video_path) as video:
            audio = video.audio
            if audio:
                audio.write_audiofile(audio_path)
                audio.close()  # Ensure proper cleanup
    except Exception as e:
        print(f"Error: {e}")


def audio_to_text_assemblyai(audio_path):
    print("Calling transcribe from AssemblyAI...")

    aai.settings.api_key = api_key 
    transcriber = aai.Transcriber()
    
    transcript = transcriber.transcribe(audio_path)
    
    return transcript.text

def chunk_text(text, max_chars):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_chars:
            chunks.append(' '.join(current_chunk[:-1]))  # Add the current chunk without the last word
            current_chunk = [word]  # Start a new chunk with the current word
    if current_chunk:
        chunks.append(' '.join(current_chunk))  # Add the last chunk
    return chunks

# Function to summarize each chunk using BART
def summarize_chunk(chunk):
    try:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error occurred: {e}"

# Function to summarize the entire text in chunks using BART
def summarize_text_bart(text):
    max_chars = 1024  # Adjust the chunk size as per the model's input constraints
    text_chunks = chunk_text(text, max_chars)  # Split the text into chunks
    chunk_count = len(text_chunks)
    
    summarized_text = ""
    
    # Summarize each chunk and collect the results
    for i, chunk in enumerate(text_chunks):
        chunk_summary = summarize_chunk(chunk)
        summarized_text += f"Summary of Part {i+1}/{chunk_count}:\n{chunk_summary}\n\n"

    return summarized_text

def save_summary_as_pdf(summary, output_pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set title font
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Text Summary", ln=True, align='C')

    # Set content font
    pdf.ln(10)  # Add a line break
    pdf.set_font("Arial", size=12)
    
    # Add summary text to the PDF
    pdf.multi_cell(0, 10, summary)
    
    # Save PDF
    pdf.output(output_pdf_path)

def download_and_process_video(url, output_folder):
    try:
        clear_download_folder(output_folder)
        # Step 1: Download YouTube video
        output_path=r'D:\Final_YTube\Downloads'
        download_youtube_video(url, output_path)

        video_file=find_first_mp4(output_folder)
        # Step 3: Extract audio from the video
        audio_file = f"{output_folder}/audio.mp3"
        extract_audio_from_video(video_file, audio_file)

        # Step 4: Convert audio to text using AssemblyAI
        transcript = audio_to_text_assemblyai(audio_file)

        # Step 5: Summarize the text in chunks using BART
        summarized_text = summarize_text_bart(transcript)

        # Step 6: Save the summary as a PDF
        summary_folder_path=r"D:\Final_YTube\output_pdf"
        clear_download_folder(summary_folder_path)
        pdf_output_path = f"{summary_folder_path}/summary.pdf"
        save_summary_as_pdf(summarized_text, pdf_output_path)

        print(f"Summary saved as PDF: {pdf_output_path}")
        return summarized_text
    except Exception as e:
        print(f"An error occurred: {e}")


def get_pdf_text(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: The file at '{pdf_path}' does not exist.")

    with open(pdf_path, "rb") as pdf:
        pdf_reader = PdfReader(pdf)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])  # Handle None values

    if not text.strip():
        raise ValueError(f"Error: No text extracted from '{pdf_path}'. Check the PDF content.")

    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to generate important questions from the extracted text of the PDF
def generate_important_questions(text, num_keywords=5):
    important_questions = []
    sentences = [s.strip() for s in text.split('.') if s.strip()]  # Remove empty sentences
    for sentence in sentences[:num_keywords]:  
        question = f"What is the importance of {sentence}?"
        important_questions.append(question)
    
    return important_questions


# Function to process PDF and generate answers for important questions


def process_pdf_and_generate_answers(pdf_path):
    # Extract text from the PDF
    pdf_text = get_pdf_text(pdf_path)

    # Generate important questions from the extracted PDF text
    important_questions = generate_important_questions(pdf_text)

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if FAISS index exists
    if not os.path.exists("faiss_index/index.faiss"):
        print("FAISS index not found. Creating a new index...")
        text_chunks = get_text_chunks(pdf_text)  # Create text chunks
        get_vector_store(text_chunks)  # Save FAISS index

    # Now load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Load the conversational chain
    chain = get_conversational_chain()

    # Iterate through the important questions and generate answers
    answers = []
    for question in important_questions:
        docs = new_db.similarity_search(question)  # Get relevant docs for the question
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answers.append({"question": question, "answer": response["output_text"]})

    return answers

# clear_download_folder("D:\Final_YTube\Downloads")
# download_youtube_video("https://www.youtube.com/watch?v=Y8Tko2YC5hA&ab_channel=ProgrammingwithMosh","D:\Final_YTube\Downloads")
# video_path=find_first_mp4("D:\Final_YTube\Downloads")
# # print(video_path)
# print(extract_audio_from_video(video_path, r"D:\Final_YTube\Downloads\audio.mp3"))
# text =audio_to_text_assemblyai(r"D:\Final_YTube\Downloads\audio.mp3")
# # print(text)
# summary=summarize_text_bart(text)
# # print(summary)
# save_summary_as_pdf(summary,r"D:\Final_YTube\output_pdf\summary1.pdf")

# print(faiss_index)
# print(embeddings)

# pdf_path = "D:\\Final_YTube\\output_pdf\\summary1.pdf"


# answers = process_pdf_and_generate_answers(pdf_path)

# # Output the generated important questions and their answers
# for idx, item in enumerate(answers, 1):
#     print(f"Q{idx}: {item['question']}")
#     print(f"A{idx}: {item['answer']}")
#     print("------------")

folder_path =r"D:\Final_YTube\Downloads"
video_url = "https://www.youtube.com/watch?v=bdUqQidffPE&ab_channel=CodeMonkey-CodingGamesforKids"
download_and_process_video(video_url, folder_path)