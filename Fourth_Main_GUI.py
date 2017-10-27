###
#Developed by Muhammad Salman and Maqsood Ahmed students of NED University of Engineering and Technology karachi, Pakistan
###

#GUI for our BCI app which calls all three classes based on condition

import threading
import os

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.logger import Logger
from kivy.core.window import Window

checker = 0

Builder.load_string("""
#:import SwapTransition kivy.uix.screenmanager.SwapTransition

<ProfileScreen>:
    id: pscreen
    text: ''
    canvas:
        Rectangle:
            source: 'profile.jpg'
            size: self.size
            pos:self.pos
    BoxLayout:
        id: profile_screen_box
        orientation: 'vertical'
        size: root.size
        size_hint: .8,1
        pos_hint: {'x':.3}
        spacing: 20
        padding: 100
        
        TextInput:
            id: create_profile_textbox
            multiline: False
            hint_text: "Create Profile"
            font_size: 20
            
        Button:
            text: 'Create Profile'
            size_hint: .4,1
            pos_hint: {'x':.3}
            on_press:root.manager.transition.direction = 'left'
            on_release: root.create_profile_func(create_profile_textbox.text)
            
        Label:
            id: lbl1
            text: 'OR'
            font_size: 50
            
        TextInput:
            id: load_profile_textbox
            multiline: False
            hint_text: "Load Profile"
            font_size: 20
            
            
        Button:
            text: 'Load Profile'
            size_hint: .4,1
            pos_hint: {'x':.3}
            on_press:root.manager.transition.direction = 'left'
            on_release: root.load_profile_func(load_profile_textbox.text)

        Label:
            id: lbl2
            text: root.text
            color:1,0,0,1
            font_size: 30


<TrainScreen>:
    id: tscreen
    text : 'machine-learning.png'
    lbl3:lbl3
    train_button: train_button

    canvas:
        Rectangle:
            id: car_source
            source: root.text
            size: self.size
            pos:self.pos
    BoxLayout:
        orientation: 'horizontal'
        Label:
            id: weather_info
            color: 0.4, 0.4, 0.4, 1
            font_size: '72sp'
            size_hint: None, None
            size: self.texture_size

    BoxLayout:
        orientation: 'horizontal'
        train_button: train_button
        Button:
            id: train_button
            text: 'Train'
            size_hint: .3,.15
            on_release: root.start_training_inside_thread(lbl3.text)

        Label:
            id: lbl3
            text: 'Start Now'
            pos_hint: {'y':-.4}
            color: .2823,.7803,1,.83
            font_size: 30
            
        Button:
            text: 'Live'
            size_hint: .3,.15
            on_press: root.change_lbl3()
            on_release: root.build_model_and_predict()
            
<PredictScreen>:
    lbl4:lbl4
    id:predictscreen
    text : ''
    canvas:
        Rectangle:
            source: root.text
            size: self.size
            pos:self.pos
    BoxLayout:
        orientation: 'vertical'
        size: root.size
        spacing: 30
        padding: 200
        
        Label:
            id: lbl4
            text: 
            font_size:30
            color:.2823,.7803,1,.83
            

        Button:
            text: "Start"
            size_hint: .3,.8
            pos_hint:{'x':-.4}
            on_press: root.start_testing_inside_thread(lbl4.text)           
            
        Button:
            text: "Shutdown"
            pos_hint:{'x':-.4}
            size_hint: .3,.8
            on_press: root.close_app()
        

""")

profile = ''

# Declare both screens
class ProfileScreen(Screen):
    
    def create_profile_func(self, profilename):
        global profile_name

        profile_name = str(profilename)

        files = os.listdir(os.getcwd()+'//UsersData//')
        check = 0

        if profilename=='':
            print('Please provide username')
            self.text = 'Please provide username'
            check = 1
            
        else:
            for f in files:
                if f==(profile_name+'.csv'):
                    check = 1
                    print('Profile already exist')
                    self.text = 'Profile already exist'
                    break
                
        if check == 0:
            self.manager.current= 'train'
        
        
    def load_profile_func(self, profilename):
        global profile_name
        profile_name = str(profilename)

        files = os.listdir(os.getcwd()+'//UsersData//')
        check = 0

        if profilename=='':
            print('Please provide username')
            self.text = 'Please provide username'
            check = 0
            
        else:
            for f in files:
                if f==(profile_name+'.csv'):
                    print('Loading your Profile')
                    check = 1
                    break
                else:
                    print('Not Loaded')
                    self.text = 'Profile Not Found'
                    check = 0
                    
            if check == 1:
                self.manager.current = 'train'

            

class TrainScreen(Screen):

    stop = threading.Event()
    
    def start_training_inside_thread(self, l_text):
        self.train_button.disabled = True
        threading.Thread(target=self.training, args=(self,l_text,)).start()

    def training(self,self_kivy, *args):
        checker = 1
        from First_CollectData import CollectData
        self.cd = CollectData(self_kivy,(os.getcwd()+'//UsersData//'),profile_name)
        self.cd.data_acq()

    def change_lbl3(self):
        self.lbl3.color = 0.32941176470588235,0.9803921568627451,0.25882352941176473,0.93 #set color
        if checker == 1:
            self.cd.disconnect()
        self.lbl3.text = 'Creating a Model'

    def build_model_and_predict(self):
        
        if checker == 1:
            self.cd.disconnect()
        from Second_ProcessData import ProcessData
    
        print(os.getcwd())
        model = ProcessData((os.getcwd()+'//UsersData//'),profile_name)
        model.main_process()
        
        self.manager.current = 'predict'

class PredictScreen(Screen):
    stop = threading.Event()
    
    def start_testing_inside_thread(self, l_text):
        threading.Thread(target=self.live_prediction, args=(self,l_text,)).start()
        
    def live_prediction(self,*args):
        from Third_Live import LivePrediction
        self.cd = LivePrediction(self,(os.getcwd()+'//UsersData//'),profile_name)
        self.cd.data_acq()

        
    def close_app(self):
        App.get_running_app().stop()
        Window.close()

# Create the screen manager
sm = ScreenManager()
sm.add_widget(ProfileScreen(name='profile'))
sm.add_widget(TrainScreen(name='train'))
sm.add_widget(PredictScreen(name='predict'))


class TestApp(App):

    def on_stop(self):
        Logger.critical('App:I\'m dying!')
        
    def build(self):
        return sm
    
if __name__ == '__main__':
    TestApp().run()
