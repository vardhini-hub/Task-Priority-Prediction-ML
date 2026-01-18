import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import matplotlib.pyplot as plt


tasks = pd.DataFrame(columns=['description', 'priority'])

try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass


def save_tasks():
    tasks.to_csv('tasks.csv', index=False)


vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)

if not tasks.empty:
    model.fit(tasks['description'], tasks['priority'])



def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({
        'description': [description],
        'priority': [priority]
    })
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()


def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()


def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)


def recommend_task():
    if not tasks.empty:
        high_priority = tasks[tasks['priority'] == 'High']

        if not high_priority.empty:
            task = random.choice(high_priority['description'].tolist())
            print(f"Recommended Task → {task} (High Priority)")
        else:
            print("No high-priority tasks available.")
    else:
        print("No tasks available.")



def visualize_tasks():
    if tasks.empty:
        print("No tasks to visualize.")
        return

    counts = tasks['priority'].value_counts()

    plt.figure()
    counts.plot(kind='bar')
    plt.title("Task Priority Distribution")
    plt.xlabel("Priority")
    plt.ylabel("Number of Tasks")
    plt.tight_layout()
    plt.show()



while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Visualize Tasks 📊")
    print("6. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        visualize_tasks()

    elif choice == "6":
        print("Goodbye!")
        break

    else:
        print("Invalid option.")
