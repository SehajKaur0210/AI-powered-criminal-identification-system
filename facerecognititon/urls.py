from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views
from .views import FileView

urlpatterns = [
    path('', views.index),  # Replaced url() with path()
    path('success', views.success),  # Replaced url() with path()

    path('add_citizen', views.addCitizen),  # Replaced url() with path()
    path('save_citizen', views.saveCitizen),  # Replaced url() with path()
    path('view_citizens', views.viewCitizens),  # Replaced url() with path()
    path('help/', views.help, name='help'),
    path('wanted_citizen/<int:citizen_id>/', views.wantedCitizen, name='wanted_citizen'),
    path('free_citizen/<int:citizen_id>/', views.freeCitizen, name='free_citizen'),

    path('login', views.login),  # Replaced url() with path()
    path('logout', views.logOut),  # Replaced url() with path()
    path('detectImage', views.detectImage),  # Replaced url() with path()
    path('detectWithWebcam', views.detectWithWebcam),  # Replaced url() with path()
    path('upload', FileView.as_view(), name='file-upload'),  # Replaced url() with path()

    path('spotted_criminals', views.spottedCriminals),  # Replaced url() with path()
    path('found_thief/<int:thief_id>/', views.foundThief, name='found_thief'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
