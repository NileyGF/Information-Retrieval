from tkinter import *
  
root = Tk()
root.mainloop()
root.title('Information Retrieval')
root.geometry('300x400')
background_color = '#001f3f'
root.configure(bg=background_color)
root.iconbitmap('logo.ico')

title = Label(root, text="This is my app", 
               bg=background_color, 
               fg='#ffffff', 
               font='android 20 bold')
title.pack()

my_button = Button(root, text='click me')
my_button.pack()

my_button = Button(root, text='click me', 
                   width=10, 
                   height=2, 
                   bg='#000000', 
                   fg='white', 
                   font='arial 20')

my_entry = Entry(root)
my_entry.pack()
my_entry = Entry(root, width=30, font='none 12', fg='red')

def print_text():
    if my_entry.get():
        text = my_entry.get()
        text_label = Label(root, text=text,
                           bg=background_color,
                           fg='#ffffff',

                           font='none 20 bold')
        text_label.pack()
        
image = PhotoImage(file='insta.gif')
image_lab = Label(root, image=image)
image_lab.pack()


