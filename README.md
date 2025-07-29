# Installation and Setup Guide

## Step 1: Install Required Libraries

Run the following command to install the necessary libraries:

pip install langchain langchain-ollama langchain-community chromadb -i https://pypi.tuna.tsinghua.edu.cn/simple/


## Step 2: Prepare `example.txt` File

Create a file named `example.txt` in your desired directory, and add the following content to it:

Kubernetes is an open-source system used for automating the deployment, scaling, and management of containerized applications. It organizes containers that constitute an application into logical units to facilitate easy management and discovery.


## Step 3: Install Ollama in WSL2

Follow these instructions to install Ollama on your WSL2 system.

## Step 4: Install UV on Windows

Follow the instructions to install UV on your Windows system.

## Step 5: Configure Port Proxy

Run the following command to configure the port proxy:

netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=11434 connectaddress=your-WSL2-private-IP connectport=11434


Replace `your-WSL2-private-IP` with your WSL2's private IP address.

## Step 6: Activate Your Virtual Environment

Activate your virtual environment to ensure you are using the correct Python version and dependencies.

## Step 7: Run the Application

Execute the following command to run your Python application:

uv run .\main.py


---

## Question: What is Kubernetes?

**Answer:**
Kubernetes is an open-source system used for automating the deployment, scaling, and management of containerized applications. It organizes containers that constitute an application into logical units to facilitate easy management and discovery.
