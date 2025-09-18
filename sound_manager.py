import pygame

pygame.mixer.init()
pygame.mixer.set_num_channels(2)

CHANNEL_YOLO1 = pygame.mixer.Channel(0)
CHANNEL_YOLO2 = pygame.mixer.Channel(1)

sound_yolo1 = pygame.mixer.Sound("music.mp3")  
sound_yolo2 = pygame.mixer.Sound("music.mp3")  