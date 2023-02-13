#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import math

import numpy as np
import pyaudio
import pygame

from tensorflow import keras
from keras.models import Model, load_model
from keras import backend as K

import mido
import pandas as pd

df = pd.read_csv('csv/meteorological.csv')
list = df.to_numpy()
normalized_list = np.array([])
steps = 0

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

for i in range(list.shape[1]-2):

    result = min_max(list[:,i+2])
    result = result - 0.5

    if i>0:
        normalized_list = np.vstack((normalized_list, result))
    else:
        normalized_list = np.append(normalized_list, result)

normalized_list = normalized_list.T

dir_name = 'results/history/'
sub_dir_name = 'e1/'
sample_rate = 48000
note_dt = 2000
note_duration = 20000
note_decay = 5.0 / sample_rate
num_params = 120
num_measures = 16
num_sigmas = 5.0
note_threshold = 32
use_pca = True
is_ae = True

background_color = (210, 210, 210)
edge_color = (60, 60, 60)
slider_colors = [(90, 20, 20), (90, 90, 20), (20, 90, 20), (20, 90, 90), (20, 20, 90), (90, 20, 90)]

note_w = 96
note_h = 96

slider_num = min(5, num_params)

control_num = 2
control_inits = [0.75, 0.5, 0.5]
control_colors = [(255, 128, 0), (0, 0, 255)]

window_width = 800
window_height = 600
margin = 20
sliders_width = int(window_width * (2.0/4.0))
sliders_height = int(window_height * (2.0/3.0))
slider_width = int((sliders_width-margin*2) / 5.0)
slider_height = sliders_height-margin*2

controls_width = int(window_width * (2.0/4.0))
controls_height = int(window_height * (1.0/3.0))
control_width = controls_width - margin*2
control_height = int((controls_height-margin*2) / 2.0)
cur_control_iy = 0
detected_keys = []
prev_measure_ix = 0

filename = 'knob.png'
knob = pygame.image.load(filename)
filename = 'control_knob.png'
c_knob = pygame.image.load(filename)
filename = 'button.png'
button_png = pygame.image.load(filename)

prev_mouse_pos = None
mouse_pressed = 0
cur_slider_ix = 0
cur_control_ix = 0
cur_control_iy = 0

volume = 3000
instrument = 0
needs_update = True
current_params = np.zeros((num_params,), dtype=np.float32)
current_notes = np.zeros((num_measures, note_h, note_w), dtype=np.uint8)
cur_controls = np.array(control_inits, dtype=np.float32)
songs_loaded = False

FUND_PITCH = 48
sound_bank = []
dev_cnt = 0
flag = 0
flag_midi_reset = 0
sr = 44100
sonification_mode = False

audio = pyaudio.PyAudio()
audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False


def audio_callback(in_data, frame_count, time_info, status):
    global audio_time
    global audio_notes
    global audio_reset
    global note_time
    global note_time_dt

    global sr, tar_sr
    global sound_bank
    global dev_cnt
    global flag
    global flag_midi_reset
    global sonification_mode
    global detected_keys
    keys = []
    global prev_measure_ix

    if audio_reset:
        audio_notes = []
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_reset = False

    if audio_pause and status is not None:
        data = np.zeros((frame_count,), dtype=np.float32)
        return data.tobytes(), pyaudio.paContinue

    cur_dt = note_dt

    while note_time_dt < audio_time + frame_count:
        measure_ix = int(note_time / note_h)

        if measure_ix >= num_measures:
            break
        note_ix = note_time % note_h

        if note_ix%24 == 0:
            if sonification_mode == True:
                update_with_sonification()

        notes = np.where(current_notes[measure_ix, note_ix] >= note_threshold)[0]
        
        for note in notes:
            freq = note + 16
            audio_notes.append((note_time_dt, freq))
            keys.append(freq)
            flag = 1

        note_time += 1
        note_time_dt += cur_dt

    if len(keys) != 0:
        detected_keys = detect_keys(keys)

    data = np.zeros((frame_count,), dtype=np.float32)
    if flag == 1:
        port = mido.open_output('IAC Driver Bus 1')

        if flag_midi_reset == 1:
            flag_midi_reset = 0
            port.reset()

        for t, f in audio_notes:
            x = np.arange(audio_time - t, audio_time + frame_count - t)
            x = np.maximum(x, 0)

            semitones = int(FUND_PITCH-f)
            tar_sr = sr*(math.pow(2.0, semitones/12.0))

            msg_on = mido.Message('note_on', note=f)
            port.send(msg_on)
            
            flag = 0
            flag_midi_reset = 1

    audio_time += frame_count
    audio_notes = [(t, f) for t, f in audio_notes if audio_time < t + note_duration]

    if note_time / note_h >= num_measures:
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_notes = []

    return data.tobytes(), pyaudio.paContinue


def update_mouse_click(mouse_pos):
    global cur_slider_ix
    global cur_control_ix
    global mouse_pressed

    global cur_control_iy
    global audio_pause
    global sonification_mode

    if margin <= mouse_pos[0] < margin+slider_width*5 and margin <= mouse_pos[1] < margin+slider_height:
        cur_slider_ix = int((mouse_pos[0]-margin*2) / slider_width)
        mouse_pressed = 1

    if margin*2 <= mouse_pos[0] < margin*2+control_width and ((sliders_height + margin*2 <= mouse_pos[1] < sliders_height + margin*2 + (control_height/2)) or (sliders_height + margin*2 + control_height <= mouse_pos[1] < sliders_height + margin*2 + control_height + (control_height/2))):
        cur_control_iy = int((mouse_pos[1] - (sliders_height + margin*2)) / (control_height))
        mouse_pressed = 2

    x = window_width*(2.0/4.0)+margin
    y = margin*2
    if x <= mouse_pos[0] < x+window_width*(2.0/4.0)-margin*3 and y <= mouse_pos[1] < y + window_height*(1/3.0)-margin*2:
        audio_pause = not audio_pause

    x = window_width*(2.0/4.0)+margin
    y = window_height*(1 / 3.0)+margin*2
    if x <= mouse_pos[0] < x+window_width*(2.0/4.0)-margin*3 and y <= mouse_pos[1] < y + window_height*(1/3.0)-margin*2:
        sonification_mode = not sonification_mode

def apply_controls():
    global note_threshold
    global note_dt
    global volume

    note_threshold = (1.0 - cur_controls[0]) * 200 + 10
    note_dt = (1.0 - cur_controls[1]) * 1800 + 200

def update_mouse_move(mouse_pos):
    global needs_update

    if mouse_pressed == 1:
        if margin <= mouse_pos[1] <= margin+slider_height:
            val = (float(mouse_pos[1]-margin) / slider_height - 0.5) * (num_sigmas * 2)
            current_params[int(cur_slider_ix)] = val
            needs_update = True
    elif mouse_pressed == 2:
        if margin <= mouse_pos[0] <= margin+control_width:
            val = float(mouse_pos[0] - margin) / control_width
            cur_controls[int(cur_control_iy)] = val
            apply_controls()

def update_with_sonification():
    global needs_update
    global steps

    for i in range(5):
        current_params[i] = float(normalized_list[steps][i]) * 10.0 * (2.5 / 5.0)
        needs_update = True

    steps = steps + 1
    if steps == list.shape[0]:
        steps = 0

def draw_controls(screen):
    global c_knob
    c_knob = pygame.transform.scale(c_knob, (30, 40))

    slider_color = (100, 100, 100)
    slider_color_layer = (195, 195, 195)

    for i in range(control_num):
        x = margin + slider_width / 2 + 5
        y = sliders_height + margin*2 + i * control_height
        w = control_width - margin*3 - 5
        h = int(control_height / 2.0)
        col = control_colors[i]

        pygame.draw.line(screen, slider_color_layer,
                         (x-15, y+(h/2.0)), (x+w+15, y+(h/2.0)), 40)
        pygame.draw.line(screen, (75, 75, 75),
                         (x, y+(h/2.0)), (x+w, y+(h/2.0)), 4)

        pygame.draw.line(screen, col,
                         (x, y+(h/2.0)), (x+int(w * cur_controls[i])-15, y+(h/2.0)), 4)
        screen.blit(c_knob, (x+int(w * cur_controls[i])-15, y))
        

def draw_sliders(screen):
    global knob
    knob = pygame.transform.scale(knob, (30, 50))

    for i in range(slider_num):
        slider_color = (100, 100, 100)
        slider_color_layer = (195, 195, 195)
        x = margin + i * slider_width
        y = margin*2

        cx = x + slider_width / 2
        cy_start = y
        cy_end = y + slider_height
        pygame.draw.line(screen, slider_color_layer, (cx, cy_start), (cx, cy_end), 16)
        pygame.draw.circle(screen, slider_color_layer, (cx+1, cy_start), 8)
        pygame.draw.circle(screen, slider_color_layer, (cx+1, cy_end), 8)
        pygame.draw.line(screen, slider_color, (cx, cy_start), (cx, cy_end), 4)

        cx_1 = x + int(slider_width* (3.0/4.0))
        cx_2 = x + slider_width-int(slider_width * (0.75/4.0))
        for j in range(int(num_sigmas * 2 + 1)):
            ly = y + slider_height / 2.0 + (j - num_sigmas) * slider_height / (num_sigmas * 2.0)
            ly = int(ly)
            col = (0, 0, 0) if j - num_sigmas == 0 else slider_color
            pygame.draw.line(screen, col, (cx_1, ly), (cx_2, ly), 1)

        py = y + int((current_params[i] / (num_sigmas * 2) + 0.5) * slider_height) - 25
        screen.blit(knob, (int(cx-15), int(py)))

def draw_button(screen):

    global button_png
    button_play = pygame.transform.scale(
        button_png, (int(window_width*(2.0/4.0)-margin*2), int(window_height*(1 / 3.0)-margin*2)))
    button_change_mode = pygame.transform.scale(
        button_png, (int(window_width*(2.0/4.0)-margin*2), int(window_height*(1 / 3.0)-margin*2)))

    screen.blit(button_play, (window_width*(2.0/4.0)+margin, margin*2))
    screen.blit(button_change_mode, (window_width*(2.0/4.0) +
                margin, window_height*(1 / 3.0)+margin*2))

def text_background(screen):
    text_background_color = (195, 195, 195)
    x = window_width*(2.0/4.0)+margin
    y = sliders_height + margin*2
    w = int(window_width*(2.0/4.0)-margin*2)
    h = int(control_height*2-margin*2)
    background_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, text_background_color, background_rect)

def draw_text(screen):

    global detected_keys

    pygame.font.init()
    font = pygame.font.SysFont(None, 50)
    description_font = pygame.font.SysFont(None, 25)
    label_font = pygame.font.SysFont(None, 15)

    text_sliders = label_font.render('LATENT VALUES (TOP 5)', True, (0, 0, 0))
    screen.blit(text_sliders, (margin*2.5, margin-5))

    text_threshold = label_font.render('THRESHOLD', True, (0, 0, 0))
    screen.blit(text_threshold, (margin*2.5, sliders_height + margin+5))

    text_speed = label_font.render('SPEED', True, (0, 0, 0))
    screen.blit(text_speed, (margin*2.5, sliders_height + margin+5 + control_height))


    for i in range(slider_num):
        x = margin + i * slider_width
        y = margin*2
        cx_2 = x + slider_width

        y1 = y + slider_height / 2.0 + (10 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y1 = int(y1)
        text_slider_value_5 = label_font.render('-5', True, (0, 0, 0))
        text_height = (text_slider_value_5.get_rect().height) / 2.0
        screen.blit(text_slider_value_5, (cx_2, y1-text_height))

        y1 = y + slider_height / 2.0 + (5 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y1 = int(y1)
        text_slider_value_0 = label_font.render('0', True, (0, 0, 0))
        text_height = (text_slider_value_0.get_rect().height) / 2.0
        screen.blit(text_slider_value_0, (cx_2, y1-text_height))

        y2 = y + slider_height / 2.0 + (0 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y2 = int(y2)
        text_slider_value_m5 = label_font.render('5', True, (0, 0, 0))
        text_height = (text_slider_value_m5.get_rect().height) / 2.0
        screen.blit(text_slider_value_m5, (cx_2, y2-text_height))


    text_play = font.render('PLAY', True, (0, 0, 0))
    text_width = (text_play.get_rect().width + margin) / 2.0
    text_height = (text_play.get_rect().height - margin) / 2.0
    screen.blit(text_play, (window_width * (3.0/4.0) - text_width,
                window_height * (1.0 / 6.0) + margin/2 - text_height))

    text_mode = font.render('MODE', True, (0, 0, 0))
    text_width = (text_mode.get_rect().width + margin) / 2.0
    text_height = (text_mode.get_rect().height - margin) / 2.0
    screen.blit(text_mode, (window_width * (3.0/4.0) - text_width,
                            window_height * (3.0 / 6.0) + margin/2 - text_height))

    text_keys = description_font.render(list[steps, 1], True, (0, 0, 0))
    text_width = (text_mode.get_rect().width + margin) / 2.0
    text_height = (text_mode.get_rect().height - margin) / 2.0
    screen.blit(text_keys, (window_width*(2.0/4.0)+margin,
                            sliders_height + margin*2))

    text_keys = description_font.render(' '.join(detected_keys), True, (0, 0, 0))
    text_width = (text_mode.get_rect().width + margin) / 2.0
    text_height = (text_mode.get_rect().height - margin) / 2.0
    screen.blit(text_keys, (window_width*(2.0/4.0)+margin,
                            sliders_height + margin*4))

def detect_keys(keys):
    result_keys = []
    semitone_dic = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
                    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}

    for key in keys:
        semitone = key%12
        octave = str(key//12 - 2)
        if semitone in semitone_dic:
            semitone = semitone_dic[semitone]
        result_keys.append(semitone+octave)
    return result_keys

def play():
    global mouse_pressed
    global current_notes
    global audio_pause
    global needs_update
    global current_params
    global prev_mouse_pos
    global audio_reset
    global instrument
    global songs_loaded
    global sonification_mode
    global steps

    print("Keras version: " + keras.__version__)

    K.set_image_data_format('channels_first')

    print("Loading encoder...")
    model = load_model(dir_name + 'model.h5')
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    decoder = K.function([model.get_layer('decoder').input, K.learning_phase()],
                         [model.layers[-1].output])

    print("Loading gaussian/pca statistics...")
    latent_means = np.load(dir_name + sub_dir_name + '/latent_means.npy')
    latent_stds = np.load(dir_name + sub_dir_name + '/latent_stds.npy')
    latent_pca_values = np.load(dir_name + sub_dir_name + '/latent_pca_values.npy')
    latent_pca_vectors = np.load(dir_name + sub_dir_name + '/latent_pca_vectors.npy')

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((int(window_width), int(window_height)))
    pygame.display.set_caption('')

    audio_stream = audio.open(
        format=audio.get_format_from_width(2),
        channels=1,
        rate=sample_rate,
        output=True,
        stream_callback=audio_callback)
    audio_stream.start_stream()

    running = True
    random_song_ix = 0
    cur_len = 0
    apply_controls()

    while running:
        # process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    prev_mouse_pos = pygame.mouse.get_pos()
                    update_mouse_click(prev_mouse_pos)
                    update_mouse_move(prev_mouse_pos)
                elif pygame.mouse.get_pressed()[2]:
                    current_params = np.zeros((num_params,), dtype=np.float32)
                    needs_update = True

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_pressed = 0
                prev_mouse_pos = None

            elif event.type == pygame.MOUSEMOTION and mouse_pressed > 0:
                update_mouse_move(pygame.mouse.get_pos())

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:

                    if not songs_loaded:
                        print("Loading songs...")
                        try:
                            y_samples = np.load('data/interim/samples.npy')
                            y_lengths = np.load('data/interim/lengths.npy')
                            songs_loaded = True
                        except Exception as e:
                            print("This functionality is to check if the model training went well by reproducing an original song. "
                                  "The composer could not load samples and lengths from model training. "
                                  "If you have the midi files, the model was trained with, process them by using"
                                  " the preprocess_songs.py to find the requested files in data/interim "
                                  "(Load exception: {0}".format(e))

                    if songs_loaded:
                        print("Random Song Index: " + str(random_song_ix))
                        if is_ae:
                            example_song = y_samples[cur_len:cur_len + num_measures]
                            current_notes = example_song * 255
                            latent_x = encoder.predict(np.expand_dims(example_song, 0), batch_size=1)[0]
                            cur_len += y_lengths[random_song_ix]
                            random_song_ix += 1
                        else:
                            random_song_ix = np.array([random_song_ix], dtype=np.int64)
                            latent_x = encoder.predict(random_song_ix, batch_size=1)[0]
                            random_song_ix = (random_song_ix + 1) % model.layers[0].input_dim

                        if use_pca:
                            current_params = np.dot(latent_x - latent_means, latent_pca_vectors.T) / latent_pca_values
                        else:
                            current_params = (latent_x - latent_means) / latent_stds

                        needs_update = True
                        audio_reset = True


        if needs_update:
            if use_pca:
                latent_x = latent_means + np.dot(current_params * latent_pca_values, latent_pca_vectors)
            else:
                latent_x = latent_means + latent_stds * current_params
            latent_x = np.expand_dims(latent_x, axis=0)
            y = decoder([latent_x, 0])[0][0]
            current_notes = (y * 255.0).astype(np.uint8)
            needs_update = False

        screen.fill(background_color)
        draw_sliders(screen)
        draw_controls(screen)
        draw_button(screen)
        text_background(screen)
        draw_text(screen)

        pygame.display.flip()
        pygame.time.wait(10)

    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Composer: Play and edit music of a trained model.')
    parser.add_argument('--model_path', type=str, help='The folder the model is stored in (e.g. a folder named e and a number located in results/history/).', required=True)

    args = parser.parse_args()
    sub_dir_name = args.model_path
    play()
