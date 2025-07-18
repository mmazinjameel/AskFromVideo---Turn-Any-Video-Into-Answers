from RAG import RAG_Class 

rag = RAG_Class("Gfr50f6ZBvo")  # just the video ID
answer = rag.prompt("What is the main topic of the video?")
print(answer)
