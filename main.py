import cv2
import numpy as np
import tensorflow as tf


def detection_classification(model, input_video_path, map_file_path):
    # Define background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

    # Read the map image file
    map_2d = cv2.imread(map_file_path)

    # Create VideoCapture instance for reading input video frames
    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise Exception('Unable to open the file!')

    # Define coordinates based on the given map to find the perspective transformation 
    points1 = np.array([(25, 30),
                        (1220, 30),
                    (1200, 760),
                    (5, 780)]).astype(np.float32)
    points2 = np.array([(40, 70),
                        (1000, 70),
                    (650, 704),
                    (340, 700)]).astype(np.float32)
    H1 = cv2.getPerspectiveTransform(points1, points2)

    points1 = np.array([(135, 115),
                        (910, 115),
                    (530, 20),
                    (555, 680)]).astype(np.float32)
    points2 = np.array([(150, 180),
                        (900, 180),
                    (525, 20),
                    (525, 690)]).astype(np.float32)
    H2 = cv2.getPerspectiveTransform(points1, points2)

    tx = 0
    ty = 0
    th = 3*np.pi / 180
    M = np.array([[np.cos(th),-np.sin(th),tx],
                [np.sin(th), np.cos(th),ty]])

    # Define a color for each team
    team_colors = [[255, 0, 0], [0, 0, 255]]

    # Define frame number
    frame_number=0

    # Loop over frames of the video to detect and classify players
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        team_color = team_colors[0]
        output_size = (frame.shape[1], frame.shape[0])
        warped_img = cv2.warpAffine(frame, M, output_size)
        warped_img = warped_img[130:, :]

        fgMask = backSub.apply(warped_img)
        index = (fgMask==127)
        fgMask = cv2.GaussianBlur(fgMask, (7, 7), 0)
        _, thresh = cv2.threshold(fgMask, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        fgMask = cv2.dilate(thresh, kernel, iterations=2)

        n, C, stats, centroids = cv2.connectedComponentsWithStats(fgMask);
        n, C = cv2.connectedComponents(fgMask);
        J = fgMask.copy()
        numberOfPlayers = n
        players = []
        map = map_2d.copy()
        for i in range(n):
            area = stats[i][4]
            if(area < 400 or area > 3000 or stats[i][1]+ stats[i][3]/2<20):
                fgMask[C == i] = 0
                numberOfPlayers-=1
            else:
                fgMask[fgMask == i] = 127
                indices= (C==i)
                left = stats[i][1]
                width = stats[i][3]
                y = left+width/2
                top = stats[i][0]
                height = stats[i][2]
                x = top + height/2
                if x > 50 and x < 1200 and y > 50 and y < 780:
                    player = warped_img[int(y - 24):int(y + 24), int(x - 18):int(x + 18)]
                    player = player / 255
                    player = np.expand_dims(player, axis=0)
                    team_predict = model.predict(player)
                    team = round(team_predict[0, 0])
                    if (team == 0):
                        team_color = team_colors[0]
                    else:
                        team_color = team_colors[1]
                pt = np.float32([[x+10, y+10]]).reshape(-1, 1, 2)
                pt = cv2.perspectiveTransform(pt, H1)
                x, y = pt[0][0][0], pt[0][0][1]
                pt = np.float32([[x, y]]).reshape(-1, 1, 2)
                pt = cv2.perspectiveTransform(pt, H2)
                x, y = pt[0][0][0], pt[0][0][1]
                players.append([x, y, team_color])
        for i in range(numberOfPlayers):
            x = players[i][0]
            y = players[i][1]
            color = players[i][2]
            cv2.circle(map, (int(x), int(y)), 3, color, 8)

        contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) < 400 or cv2.contourArea(cnt) >2000:
                continue
            cv2.rectangle(warped_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Add detection rectangle and frame number
        cv2.rectangle(warped_img, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(warped_img, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        cv2.imshow('Frame', warped_img)
        cv2.imshow('Map', map)
        frame_number+=1
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def main():
    map_file_path = "assets/map/2D_map.png"
    input_video_path = "assets/data/input.mp4"
    trained_model_path = "assets/trained_model/classifier_cnn.model"

    # Load classification model
    classification_model = tf.keras.models.load_model(trained_model_path)

    # Detect and classify players
    detection_classification(classification_model, input_video_path, map_file_path)


if __name__ == "__main__":
    main()
