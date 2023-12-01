import tkinter as tk

caffieneAdder = 0

def Adder(caffieneAdder):
	caffieneAdder += 1
	return caffieneAdder

# Create a window
window = tk.Tk()

# Set the window title
window.title("Caffiene Tracker")

# Set the window size
window.geometry("400x300")

# Create a frame
mainframe = tk.Frame(window)
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# Create a button
button = tk.Button(mainframe, text='+', command=Adder)
button.grid(column=3, row=3, sticky=tk.W)

# Display the window
window.mainloop()
