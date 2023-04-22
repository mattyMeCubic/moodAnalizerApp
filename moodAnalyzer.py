# import required libraries
import tkinter as tk # Tkinter for GUI
import torch # PyTorch for neural network
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Hugging Face Transformers for pre-trained BERT model
import webbrowser # to open a link in a web browser

# define a class for the GUI application
class MoodAnalyzerApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Mood Analyzer") # Set window title
        self.pack()
        self.create_widgets() # Create widgets in the GUI
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ) # Load tokenizer for BERT model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ) # Load pre-trained BERT model

    def create_widgets(self):
        # create input label
        self.input_label = tk.Label(self, text="Enter your text below:")
        self.input_label.pack()

        # create input text box
        self.input_box = tk.Text(self, height=10, width=50)
        self.input_box.pack()

        # create analyze button
        self.analyze_button = tk.Button(
            self, text="Analyze", command=self.analyze_text
        )
        self.analyze_button.pack()

        # create output label
        self.output_label = tk.Label(
            self, text="Your mood and suggestions will appear here:"
        )
        self.output_label.pack()

        # create output text box
        self.output_box = tk.Text(
            self, height=8, width=50, state=tk.DISABLED
        )
        self.output_box.pack()

        # create resource button
        self.resource_button = tk.Button(
            self,
            text="Get Mental Health Resources",
            command=self.open_resource,
        )
        self.resource_button.pack()

    def analyze_text(self):
        # get input text from text box
        input_text = self.input_box.get("1.0", "end-1c")

        # tokenize input text for BERT model
        inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )

        # pass input through BERT model to get predicted sentiment
        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits).item()

        # set mood and suggestions based on predicted sentiment
        if prediction == 0:
            self.mood = "negative"
            self.suggest_text = "Here are some resources for mental health support:"
            self.suggestion_1 = "Call the National Suicide Prevention Lifeline at 1-800-273-8255"
            self.suggestion_2 = (
                "Visit the National Alliance on Mental Illness (NAMI) "
                "website at www.nami.org"
            )
            self.suggestion_3 = (
                "Try online therapy with BetterHelp or Talkspace"
            )
        elif prediction == 1:
            self.mood = "neutral"
            self.suggest_text = "Here are some tips to maintain good mental health:"
            self.suggestion_1 = (
                "Get enough sleep and exercise regularly"
            )
            self.suggestion_2 = "Eat a balanced diet and stay hydrated"
            self.suggestion_3 = (
                "Practice mindfulness or meditation"
            )
        else:
            self.mood = "positive"
            self.suggest_text = (
                "Here are some suggestions to maintain your positive mood:"
            )
            self.suggestion_1 = (
                "Listen to your favorite music or podcast"
            )
            self.suggestion_2 = "Go for a walk or jog outside"
            self.suggestion_3 = (
                "Spend time with loved ones or friends"
            )

        # call the display_output method to update the output box with the results
        self.display_output()

def display_output(self):
    # enable the output box and clear its contents
    self.output_box.configure(state=tk.NORMAL)
    self.output_box.delete("1.0", tk.END)

    # insert the mood and suggestions into the output box
    self.output_box.insert(tk.END, f"Your mood is {self.mood}.\n\n")
    self.output_box.insert(tk.END, self.suggest_text + "\n")
    self.output_box.insert(tk.END, f"- {self.suggestion_1}\n")
    self.output_box.insert(tk.END, f"- {self.suggestion_2}\n")
    self.output_box.insert(tk.END, f"- {self.suggestion_3}\n")

    # disable the output box to prevent further editing
    self.output_box.configure(state="disabled")

def open_resource(self):
    # open the NAMI website in a new browser tab
    webbrowser.open_new("https://www.nami.org/help")