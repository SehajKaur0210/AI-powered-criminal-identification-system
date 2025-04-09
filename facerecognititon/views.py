from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
import bcrypt
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from django.contrib.auth import logout
from .models import User, Criminal, CriminalLastSpotted


class FileView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
      file_serializer.save()
      return Response(file_serializer.data, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# view for index
def index(request):
    return render(request, 'session/login.html')


#view for log in
def login(request):
    if((User.objects.filter(email=request.POST['login_email']).exists())):
        user = User.objects.filter(email=request.POST['login_email'])[0]
        if ((request.POST['login_password']== user.password)):
            request.session['id'] = user.id
            request.session['name'] = user.first_name
            request.session['surname'] = user.last_name
            messages.add_message(request,messages.INFO,'Welcome to criminal detection system '+ user.first_name+' '+user.last_name)
            return redirect(success)
        else:
            messages.error(request, 'Oops, Wrong password, please try a diffrerent one')
            return redirect('/')
    else:
        messages.error(request, 'Oops, That police ID do not exist')
        return redirect('/')


#view for log out
def logOut(request):
    logout(request)
    messages.add_message(request,messages.INFO,"Successfully logged out")
    return redirect(index)


# view to add crimina
def addCitizen(request):
   return render(request, 'home/add_citizen.html')


# view to add save citizen
def saveCitizen(request):
    if request.method == 'POST':
        citizen=Criminal.objects.filter(aadhar_no=request.POST["aadhar_no"])
        if citizen.exists():
            messages.error(request,"Citizen with that Aadhar Number already exists")
            return redirect(addCitizen)
        else:
            myfile = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)

            criminal = Criminal.objects.create(
                name=request.POST["name"],
                aadhar_no=request.POST["aadhar_no"],
                address=request.POST["address"],
                picture=uploaded_file_url[1:],
                status="Free"
            )
            criminal.save()
            messages.add_message(request, messages.INFO, "Citizen successfully added")
            return redirect(viewCitizens)


# view to get citizen(criminal) details
def viewCitizens(request):
    citizens=Criminal.objects.all()
    context={
        "citizens":citizens
    }
    return render(request,'home/view_citizens.html',context)


#view to set criminal status to wanted
def wantedCitizen(request, citizen_id):
    wanted = Criminal.objects.filter(pk=citizen_id).update(status='Wanted')
    if (wanted):
        messages.add_message(request,messages.INFO,"User successfully changed status to wanted")
    else:
        messages.error(request,"Failed to change the status of the citizen")
    return redirect(viewCitizens)

#view to set criminal status to free
def freeCitizen(request, citizen_id):
    free = Criminal.objects.filter(pk=citizen_id).update(status='Free')
    if (free):
        messages.add_message(request,messages.INFO,"User successfully changed status to Found and Free from Search")
    else:
        messages.error(request,"Failed to change the status of the citizen")
    return redirect(viewCitizens)


def spottedCriminals(request):
    thiefs=CriminalLastSpotted.objects.filter(status="Wanted")
    context={
        'thiefs':thiefs
    }
    return render(request,'home/spotted_thiefs.html',context)


def foundThief(request,thief_id):
    free = CriminalLastSpotted.objects.filter(pk=thief_id)
    freectzn = CriminalLastSpotted.objects.filter(aadhar_no=free.get().aadhar_no).update(status='Found')
    if(freectzn):
        thief = CriminalLastSpotted.objects.filter(pk=thief_id)
        free = Criminal.objects.filter(aadhar_no=thief.get().aadhar_no).update(status='Found')
        if(free):
            messages.add_message(request,messages.INFO,"Thief updated to found, congratulations")
        else:
            messages.error(request,"Failed to update thief status")
    

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import pickle
import random
import nltk
import os
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from django.shortcuts import render
from django.http import JsonResponse

# Path to your saved model and other resources
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load necessary resources for prediction
intents = json.loads(open(r'D:\group_project\crime_project28\my\intents.json').read())
words = pickle.load(open(r'D:\group_project\crime_project28\my\words.pkl','rb'))
classes = pickle.load(open(r'D:\group_project\crime_project28\my\classes.pkl','rb'))
model = load_model(r'D:\group_project\crime_project28\my\chatbott.keras')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lemmatizer = WordNetLemmatizer()

# Clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

# Convert sentence to bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent of the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in result:
        result_list.append({"intent": classes[r[0]], 'probability': str(r[1])})
    return result_list

# Get the response for the predicted intent
def get_response(intent_list, intents_json):
    tag = intent_list[0]['intent']
    intents_list = intents_json['intents']
    for intent in intents_list:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response

# The view function for handling chatbot messages
def help(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        intent_list = predict_class(message)
        response = get_response(intent_list, intents)
        return JsonResponse({'bot_reply': response})

    return render(request, 'home/help.html')

def success(request):
    user = User.objects.get(id=request.session['id'])
    context = {
        "user": user
    }
    return render(request, 'home/welcome.html', context)


def detectImage(request):
    # function to detect faces and draw a rectangle around the faces
    # with correct face label

    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

    # get the criminal id, name, images from the database
    images = []
    encodings = []
    names = []
    files = []

    prsn = Criminal.objects.all()
    for criminal in prsn:
        images.append(criminal.name + '_image')
        encodings.append(criminal.name + '_face_encoding')
        files.append(criminal.picture)
        names.append(criminal.name + ' ' + criminal.address)

    for i in range(0, len(images)):
        images[i] = face_recognition.load_image_file(files[i])
        encodings[i] = face_recognition.face_encodings(images[i])[0]

    # encoding the faces of the criminals in the database
    # creating array of their names
    known_face_encodings = encodings
    known_face_names = names

    # loading the image that is coming from the front end
    unknown_image = face_recognition.load_image_file(uploaded_file_url[1:])

    # finding face locations and encoding of that image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # converting the image to PIL format
    pil_image = Image.fromarray(unknown_image)
    # Draw a rectangle over the face
    draw = ImageDraw.Draw(pil_image)

    # run a for loop to find if faces in the input image match those 
    # in our database
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # compare the face to the criminals present
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # find distance w.r.t to the faces of criminals present in the DB
        # take the minimum distance
        # see if it matches the faces
        # if matches update the name variable to the respective criminal name
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # with Pillow module draw a rectangle around the face
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Use textbbox to calculate text size
        text_bbox = draw.textbbox((left, bottom), name, font=None)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Draw rectangle for the name label
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory 
    del draw

    # display the image 
    pil_image.show()
    return redirect('/success')


# View to detect criminals using webcam
import dlib
import cv2
import numpy as np
from django.shortcuts import redirect

from django.core.files.storage import default_storage
from django.conf import settings
import os



def detectWithWebcam(request):
        # Accessing the deafult camera of the system
    video_capture = cv2.VideoCapture(0)

    # Loading faces from DB with their data.
    images=[]
    encodings=[]
    names=[]
    files=[]
    nationalIds=[]

    prsn=Criminal.objects.all()
    for criminal in prsn:
        images.append(criminal.name+'_image')
        encodings.append(criminal.name+'_face_encoding')
        files.append(criminal.picture)
        names.append('Name: '+criminal.name+ ', AadharNo: '+ criminal.aadhar_no+', Address '+criminal.address)
        nationalIds.append(criminal.aadhar_no)

    #finding encoding of the criminals
    for i in range(0,len(images)):
        images[i]=face_recognition.load_image_file(files[i])
        encodings[i]=face_recognition.face_encodings(images[i])[0]


    # Encoding of faces and their respective ids and names
    known_face_encodings = encodings
    known_face_names = names
    n_id = nationalIds



    while True:
        # Reading a single frame of the video
        ret, frame = video_capture.read()

        # Finding all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Run a loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          
           # checking if the faces in the frame matches to that from our DB
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # finding distance of the faces in the frame to that from our DB
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            #if it matches with the one with minimum distance then print their name on the frame
            if matches[best_match_index]:
                ntnl_id = n_id[best_match_index]
                criminal = Criminal.objects.filter(aadhar_no=ntnl_id)
                name = known_face_names[best_match_index]+', Status: '+criminal.get().status


                # if the face is of a wanted criminal then add it to CriminalLastSpotted list
                if(not(criminal.get().status=='Wanted')):
                    thief = CriminalLastSpotted.objects.create(
                        name=criminal.get().name,
                        aadhar_no=criminal.get().aadhar_no,
                        address=criminal.get().address,
                        picture=criminal.get().picture,
                        status='Wanted',
                        latitude='25.3176° N',
                        longitude='82.9739° E')
                    thief.save()



            # Drawing Rectangular box around the face(s)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Put a label of their name 
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Now display their faces with frames
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break

# Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    return redirect('/success')