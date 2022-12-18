"""
Tres modelos de recuperacion de informacion utilizando tres colecciones de prueba.
"""
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import psri.Vector_model as Vector_model
import psri.LSI_model as LSI_model
import psri.Boolean_model as Boolean_model

class ProyectodeSRI(toga.App):
    b = False
    vec_cran = Vector_model.Vector_model('cranfield')
    vec_vsw = Vector_model.Vector_model('vaswani')
    vec_nfc = Vector_model.Vector_model('nfcorpus')
    bool_cran = Boolean_model.Boolean_model('cranfield')
    bool_vsw = Boolean_model.Boolean_model('vaswani')
    bool_nfc = Boolean_model.Boolean_model('nfcorpus')
    lsi_cran = LSI_model.LSI_model('cranfield')
    lsi_vsw = LSI_model.LSI_model('vaswani_r')
    lsi_nfc = LSI_model.LSI_model('nfcorpus_r')
    
    model_enviroment = {('vector', 'cranfield'):    vec_cran,
                        ('vector', 'vaswani'):      vec_vsw,
                        ('vector', 'nfcorpus'):     vec_nfc,
                        ('boolean', 'cranfield'):   bool_cran,
                        ('boolean', 'vaswani'):     bool_vsw,
                        ('boolean', 'nfcorpus'):    bool_nfc,
                        ('LSI', 'cranfield'):       lsi_cran,
                        ('LSI', 'vaswani'):         lsi_vsw,
                        ('LSI', 'nfcorpus'):        lsi_nfc
                        }
    
    def startup(self):
        """
        Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """
        main_box = toga.Box(style=Pack(direction=COLUMN))
       # main_image = toga.Image(os.path.dirname(__file__),)
        image_box = toga.Box(style=Pack(direction=COLUMN))
        main_image = toga.Image("resources/psri.png")
        main_imageview = toga.ImageView(main_image)
        main_imageview.style.update(height=72)
        image_box.add(main_imageview)
        search_lbl =toga.Label('Search',style=Pack( padding_top=2))
        search_lbl.focus()
        self.search_input = toga.TextInput(placeholder='Write here your query.', style=Pack(padding_top=2,flex=1))
        search_btm = toga.Button('Search',on_press=self.send_query, style=Pack(padding_top=1))
        
        search_box= toga.Box(style=Pack(direction=ROW, padding= 5))
        search_box.add(search_lbl)
        search_box.add(self.search_input)
        search_box.add(search_btm)
        
        interm_box= toga.Box (style=Pack(direction=ROW, padding_top=20,padding_left=5,padding_right=5))
        
        self.model_box = toga.Box (style=Pack(flex=1 ,direction=COLUMN))
        model_lbl = toga.Label('Models',style=Pack(padding=1 ,flex=0))
        
        self.vectorial_check = toga.Switch(text='Vector Space Model',style=Pack(padding_top=5),on_change=self.mcheckboxes)
        self.LSI_check = toga.Switch(text='Latent Semantic Indexing Model',style=Pack(padding_top=5),on_change=self.mcheckboxes)
        self.boolean_check = toga.Switch(text='Boolean Model',style=Pack(padding_top=5),on_change=self.mcheckboxes)
        
        self.model_box.add(model_lbl)
        self.model_box.add(self.vectorial_check)
        self.model_box.add(self.LSI_check)
        self.model_box.add(self.boolean_check)
        
        
        self.enviroment_box = toga.Box (style=Pack(flex=1, direction=COLUMN))
        enviroment_lbl = toga.Label('Enviroments',style=Pack(padding=1,flex=1))
        self.cranfield_check = toga.Switch(text='Cranfield Collection',style=Pack(padding_top=5),on_change=self.echeckboxes)
        self.vaswani_check = toga.Switch(text='Vaswani Collection',style=Pack(padding_top=5),on_change=self.echeckboxes)
        self.nfcorpus_check = toga.Switch(text='NFCorpus Collection',style=Pack(padding_top=5),on_change=self.echeckboxes)
        
        self.enviroment_box.add(enviroment_lbl)
        self.enviroment_box.add(self.cranfield_check)
        self.enviroment_box.add(self.vaswani_check)
        self.enviroment_box.add(self.nfcorpus_check)
        
        interm_box.add(self.model_box)
        interm_box.add(self.enviroment_box)
        main_box.add(image_box)
        main_box.add(search_box)
        main_box.add(interm_box)

        
        self.docs_box = toga.Box(style=Pack(direction=COLUMN, padding_top=20,padding_left=5,padding_right=5))
        self.docs_table = toga.Table(['Num','Id','Title'],missing_value=' ',on_double_click=self.open_document,style=Pack(flex=1, padding = 1))
        #self.scroller = toga.ScrollContainer( horizontal=False, style=Pack(padding=10))
        self.docs_box.add(self.docs_table)
        main_box.add(self.docs_box)
        
        #self.scroller.content = self.docs_table
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def mcheckboxes(self,widget):
        if widget.text == 'Latent Semantic Indexing Model' and not self.b:
            widget.value = self.b = True            
            self.vectorial_check.value = False
            self.boolean_check.value = False
            self.b = False
        if widget.text == 'Vector Space Model' and not self.b:
            widget.value = self.b = True
            self.LSI_check.value = False
            self.boolean_check.value = False
            self.b = False
        if widget.text == 'Boolean Model' and not self.b:
            widget.value = self.b = True
            self.LSI_check.value = False
            self.vectorial_check.value = False
            self.b = False

    def echeckboxes(self,widget):
        if widget.text == 'Cranfield Collection' and not self.b:
            widget.value = self.b = True            
            self.vaswani_check.value = False
            self.nfcorpus_check.value = False
            self.b = False
        if widget.text == 'Vaswani Collection' and not self.b:
            widget.value = self.b = True
            self.cranfield_check.value = False
            self.nfcorpus_check.value = False
            self.b = False
        if widget.text == 'NFCorpus Collection' and not self.b:
            widget.value = self.b = True
            self.cranfield_check.value = False
            self.vaswani_check.value = False
            self.b = False
    
    def send_query(self, widget):
        if not ((self.cranfield_check.value or self.vaswani_check.value or self.nfcorpus_check.value) and (self.vectorial_check.value or self.LSI_check.value or self.boolean_check.value)):
            self.main_window.info_dialog('Ups','You need to select an option from models and one from enviroments')
            return
        query_text = self.search_input.value
        
        if self.cranfield_check.value:      collection_sel = 'cranfield'
        elif self.vaswani_check.value:      collection_sel = 'vaswani'
        elif self.nfcorpus_check.value:     collection_sel = 'nfcorpus'
        else:     print("Wrong collection")
        
        if self.vectorial_check.value:      model_sel = 'vector'
        elif self.boolean_check.value:      model_sel = 'boolean'
        elif self.LSI_check.value:          model_sel = 'LSI'
        else:     print("Wrong model")
        
        model = ProyectodeSRI.model_enviroment[(model_sel, collection_sel)]
        self.current_documents = model.query(query_text, 30)
        self.show_docs(30)
                                   #ranking defines how many documents are gonna be returned
    def show_docs(self, ranking = 25):
        i=0
        while i < len(self.docs_table.data):
            self.docs_table.data.remove(self.docs_table.data[0])
        for i in range (ranking):
            if i < len (self.current_documents):
                self.docs_table.data.append(i+1,self.current_documents[i][0], self.current_documents[i][1])
                
    def open_document(self, widget, row):
        row_ind = row.num - 1 
        document = '\t\t' + self.current_documents[row_ind][1] + '\n\n' + self.current_documents[row_ind][2]
        # self.main_window.info_dialog(':v',"Working on it")
        self.document_window = toga.Window(title=f"{row.title}",closeable=True)
        self.windows.add(self.document_window)
        self.justa_scroll = toga.ScrollContainer ()
        self.justa_scroll.content=toga.Label(text=document)
        self.document_window.content = self.justa_scroll  
        self.document_window.show()
        return 

        

def main():
    return ProyectodeSRI()
