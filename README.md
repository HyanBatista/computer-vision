# Computer vision

Abaixo seguem instruções de como executar em seu computador os principais scripts desse repositório. Elas foram escritas supondo que o usuário está usando um sistema operacional unix-based.

DETECÇÃO DE FACES COM OPENCV e CNN

*python detect_faces_video_file.py -p ../res/deploy.prototxt.txt -m ../res/res10_300x300_ssd_iter_140000.caffemodel -v [file_name]*

Ao executar o comando acima, uma janela semelhante a esta aparecerá:<br>

![Detecção de faces com OpenCV](/images/detect_faces_video_file_showcase.png)

EYE BLINK DETECTION BASEADO EM FACIAL LANDMARKS

*python detect_blink.py -p ../res/shape_predictor_68_face_landmarks.dat*

Ao inserir esse comando no prompt, o output será algo parecido com isto:<br>

![Eye blink detection com dlib](/images/detect_blink_showcase.png)
