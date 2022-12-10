"""
Tres modelos de recuperacion de informacion utilizando tres colecciones de prueba.
"""
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import ir_datasets
# import backend.Vector_model
# import backend.LSI_model
import psri.Vector_model
import psri.LSI_model

cranfield = ir_datasets.load('cranfield')

class ProyectodeSRI(toga.App):
    b = False
    def startup(self):
        """
        Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """
        main_box = toga.Box(style=Pack(direction=COLUMN))
        search_lbl =toga.Label('Search',style=Pack( padding_top=2))
        search_lbl.focus()
        self.search_input = toga.TextInput(placeholder='Write here your query.', style=Pack(padding_top=2,flex=1))
        search_btm = toga.Button('Search',on_press=self.send_query, style=Pack(padding_top=1))
        
        search_box= toga.Box(style=Pack(direction=ROW, padding_top=70,padding_left=5,padding_right=5))
        search_box.add(search_lbl)
        search_box.add(self.search_input)
        search_box.add(search_btm)
        
        interm_box= toga.Box (style=Pack(direction=ROW, padding_top=20,padding_left=5,padding_right=5))
        
        self.model_box = toga.Box (style=Pack(flex=1 ,direction=COLUMN))
        model_lbl = toga.Label('Models',style=Pack(padding=1 ,flex=0))
        
        self.vectorial_check = toga.Switch(text='Vector Space Model',style=Pack(padding_top=5),on_change=self.mcheckboxes)
        self.LSI_check = toga.Switch(text='Latent Semantic Indexing Model',style=Pack(padding_top=5),on_change=self.mcheckboxes)
        self.TEMPORAL_check = toga.Switch(text='Temporal Model',style=Pack(padding_top=5),on_change=self.mcheckboxes, enabled=False)
        
        self.model_box.add(model_lbl)
        self.model_box.add(self.vectorial_check)
        self.model_box.add(self.LSI_check)
        self.model_box.add(self.TEMPORAL_check)
        
        
        self.enviroment_box = toga.Box (style=Pack(flex=1, direction=COLUMN))
        enviroment_lbl = toga.Label('Enviroments',style=Pack(padding=1,flex=1))
        self.cranfield_check = toga.Switch(text='Cranfield Collection',style=Pack(padding_top=5),on_change=self.echeckboxes)
        self.reuters_check = toga.Switch(text='Reuters-21578 Collection',style=Pack(padding_top=5),on_change=self.echeckboxes, enabled=False)
        self.newsgroups_check = toga.Switch(text='20 Newsgroups Collection',style=Pack(padding_top=5),on_change=self.echeckboxes, enabled=False)
        
        self.enviroment_box.add(enviroment_lbl)
        self.enviroment_box.add(self.cranfield_check)
        self.enviroment_box.add(self.reuters_check)
        self.enviroment_box.add(self.newsgroups_check)
        
        interm_box.add(self.model_box)
        interm_box.add(self.enviroment_box)
        main_box.add(search_box)
        main_box.add(interm_box)

        
        self.docs_box = toga.Box(style=Pack(direction=COLUMN, padding_top=20,padding_left=5,padding_right=5))
        self.scroller = toga.ScrollContainer( horizontal=False, style=Pack(padding=10))
        main_box.add(self.scroller)
        # self.scroller.content = main_box
        self.scroller.content = self.docs_box

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def mcheckboxes(self,widget):
        if widget.text == 'Latent Semantic Indexing Model' and not self.b:
            widget.value = self.b = True            
            self.vectorial_check.value = False
            self.TEMPORAL_check.value = False
            self.b = False
        if widget.text == 'Vector Space Model' and not self.b:
            widget.value = self.b = True
            self.LSI_check.value = False
            self.TEMPORAL_check.value = False
            self.b = False
        if widget.text == 'Temporal Model' and not self.b:
            widget.value = self.b = True
            self.LSI_check.value = False
            self.vectorial_check.value = False
            self.b = False

    def echeckboxes(self,widget):
        if widget.text == 'Cranfield Collection' and not self.b:
            widget.value = self.b = True            
            self.reuters_check.value = False
            self.newsgroups_check.value = False
            self.b = False
        if widget.text == 'Reuters-21578 Collection' and not self.b:
            widget.value = self.b = True
            self.cranfield_check.value = False
            self.newsgroups_check.value = False
            self.b = False
        if widget.text == '20 Newsgroups Collection' and not self.b:
            widget.value = self.b = True
            self.cranfield_check.value = False
            self.reuters_check.value = False
            self.b = False
    
    def send_query(self, widget):
        if not ((self.cranfield_check.value or self.reuters_check.value or self.newsgroups_check.value) and (self.vectorial_check.value or self.LSI_check.value or self.TEMPORAL_check.value)):
            self.main_window.info_dialog('Ups','You need to select an option from models and one from enviroments')
            return
        query = self.search_input.value
        if self.cranfield_check.value:
            collection = 'cranfield'
        elif self.reuters_check.value:
            collection = 'Reuters-21578'
        elif self.newsgroups_check.value:
            collection = '20 Newsgroups'
        else:
            print("Wrong collection")
        if self.vectorial_check.value:
            v = psri.Vector_model.Vector_model(collection)
            docs = v.query(query)
        elif self.LSI_check.value:
            l = psri.LSI_model.LSI_model(collection)
            docs = l.query(query)
        else:
            print("Wrong model")
            
        self.show_docs(docs)
        
    def show_docs(self, documents, ranking = 25): 
        i=0
        while i < len(self.docs_box.children):
            self.docs_box.remove(self.docs_box.children[i])
        i=0
        for i in range(ranking):
            if i < len(documents):
                self.docs_box.add(toga.Label(documents[i],style=Pack(padding=1)))
        self.docs_box.add(toga.Label('',style=Pack( padding_top=2)))
        self.docs_box.add(toga.Label('',style=Pack( padding_top=2)))
        # The last two blank labels are there to make the 25th element look better. NEEDS FIXING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!a


def main():
    return ProyectodeSRI()
