import os
import cv2
import uuid
import numpy as np

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ===============================
# BALL TRACKING ENGINE
# ===============================

def detect_ball(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    positions = []

    frame_number = 0
    previous = None


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        frame_number += 1


        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        # white cricket ball mask

        lower = np.array(
            [0,0,170]
        )

        upper = np.array(
            [180,60,255]
        )


        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )


        mask = cv2.medianBlur(
            mask,
            5
        )


        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        candidate=None


        for c in contours:

            area=cv2.contourArea(c)


            if area < 5 or area > 250:
                continue


            x,y,w,h=cv2.boundingRect(c)


            cx=x+w//2
            cy=y+h//2


            if previous:

                distance=np.sqrt(
                    (cx-previous[0])**2+
                    (cy-previous[1])**2
                )


                if distance>80:
                    continue


            candidate=(cx,cy)


        if candidate:

            positions.append(
                {
                    "frame":frame_number,
                    "x":candidate[0],
                    "y":candidate[1]
                }
            )

            previous=candidate



    cap.release()

    return positions



# ===============================
# TRAJECTORY ANALYSIS
# ===============================


def calculate_path(points,width,height):


    if len(points)<5:

        return {

            "pitching":"UNKNOWN",
            "impact":"UNKNOWN",
            "wicket":"UNKNOWN",
            "decision":"NOT OUT"

        }



    xs=np.array(
        [p["x"] for p in points]
    )

    ys=np.array(
        [p["y"] for p in points]
    )


    # bounce point

    bounce_index=np.argmax(ys)


    pitch_x=float(
        xs[bounce_index]
    )

    pitch_y=float(
        ys[bounce_index]
    )



    # impact estimate

    impact_index=min(
        bounce_index+5,
        len(xs)-1
    )


    impact_x=float(
        xs[impact_index]
    )

    impact_y=float(
        ys[impact_index]
    )



    # wicket projection

    coeff=np.polyfit(
        ys,
        xs,
        2
    )


    stump_y=height*0.45


    projected_x=float(
        np.polyval(
            coeff,
            stump_y
        )
    )


    stump_left=width*0.46
    stump_right=width*0.54



    if projected_x >= stump_left and projected_x <= stump_right:

        decision="OUT"
        wicket="HITTING"


    else:

        decision="NOT OUT"
        wicket="MISSING"



    return {


        "pitching":"IN LINE",

        "impact":"IN LINE",

        "wicket":wicket,

        "decision":decision,


        "pitch":{
            "x":pitch_x,
            "y":pitch_y
        },


        "impact_point":{
            "x":impact_x,
            "y":impact_y
        },


        "projection":{
            "x":projected_x,
            "y":stump_y
        }


    }



# ===============================
# ROUTES
# ===============================


@app.route("/")
def home():

    return render_template(
        "index.html"
    )



@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():


    if "video" not in request.files:

        return jsonify(
            {
                "error":"No video"
            }
        )


    file=request.files["video"]


    filename=str(uuid.uuid4()) + os.path.splitext(
        secure_filename(file.filename)
    )[1]


    path=os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    file.save(path)



    points=detect_ball(path)



    cap=cv2.VideoCapture(path)


    width=int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height=int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    cap.release()



    result=calculate_path(
        points,
        width,
        height
    )



    return jsonify(

        {

        "success":True,

        "video":url_for(
            "static",
            filename="uploads/"+filename
        ),

        "ball_points":points,

        "analysis":result

        }

    )



if __name__=="__main__":


    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )
