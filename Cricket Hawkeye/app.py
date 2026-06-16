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

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =====================================================
# BALL DETECTION ENGINE
# Motion + HSV + Size Filtering
# =====================================================

def detect_ball(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []


    points = []

    previous_frame = None
    previous_point = None

    frame_id = 0


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        frame_id += 1


        height, width = frame.shape[:2]


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )


        candidates = []


        # -----------------------------
        # WHITE BALL MASK
        # -----------------------------

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        lower = np.array(
            [0,0,180]
        )

        upper = np.array(
            [180,70,255]
        )


        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )


        # -----------------------------
        # MOTION FILTER
        # -----------------------------

        if previous_frame is not None:


            diff = cv2.absdiff(
                previous_frame,
                blur
            )


            _,motion=cv2.threshold(
                diff,
                25,
                255,
                cv2.THRESH_BINARY
            )


            mask=cv2.bitwise_and(
                mask,
                motion
            )


        kernel=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3,3)
        )


        mask=cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )


        contours,_=cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        for c in contours:


            area=cv2.contourArea(c)


            if area < 4 or area > 120:
                continue



            x,y,w,h=cv2.boundingRect(c)


            ratio=w/h if h else 0


            if ratio < 0.5 or ratio > 2:
                continue



            cx=x+w//2
            cy=y+h//2



            # Remove impossible jumps

            if previous_point:


                distance=np.sqrt(
                    (cx-previous_point[0])**2+
                    (cy-previous_point[1])**2
                )


                if distance > 120:
                    continue



            candidates.append(
                (cx,cy,area)
            )



        if candidates:


            # choose biggest valid moving object

            candidates.sort(
                key=lambda x:x[2],
                reverse=True
            )


            point=candidates[0]


            points.append(
                {
                    "frame":frame_id,
                    "x":point[0],
                    "y":point[1]
                }
            )


            previous_point=(
                point[0],
                point[1]
            )


        previous_frame=blur



    cap.release()


    return smooth_points(points)




# =====================================================
# REMOVE BAD POINTS
# =====================================================

def smooth_points(points):


    if len(points)<3:
        return points


    cleaned=[]


    last=None


    for p in points:


        if last:


            d=np.sqrt(
                (p["x"]-last["x"])**2+
                (p["y"]-last["y"])**2
            )


            if d>150:
                continue


        cleaned.append(p)

        last=p



    return cleaned





# =====================================================
# TRAJECTORY + LBW ENGINE
# =====================================================

def analyse_lbw(points,width,height):


    if len(points)<5:


        return {

            "pitching":"UNKNOWN",
            "impact":"UNKNOWN",
            "wickets":"UNKNOWN",
            "decision":"NOT OUT",
            "points":points

        }



    xs=np.array(
        [p["x"] for p in points],
        dtype=float
    )


    ys=np.array(
        [p["y"] for p in points],
        dtype=float
    )



    # Bounce estimation

    bounce_index=np.argmax(
        ys
    )


    pitch_x=float(
        xs[bounce_index]
    )


    pitch_y=float(
        ys[bounce_index]
    )



    # Impact after bounce

    impact_index=min(
        bounce_index+4,
        len(xs)-1
    )


    impact_x=float(
        xs[impact_index]
    )


    impact_y=float(
        ys[impact_index]
    )



    # Trajectory fitting

    try:

        curve=np.polyfit(
            ys,
            xs,
            2
        )


    except:


        curve=np.polyfit(
            ys,
            xs,
            1
        )



    stump_y=height*0.45


    projected_x=float(
        np.polyval(
            curve,
            stump_y
        )
    )



    # approximate wicket zone

    stump_left=width*0.46

    stump_right=width*0.54



    if projected_x < stump_left or projected_x > stump_right:


        decision="NOT OUT"

        wickets="MISSING"


    else:


        decision="OUT"

        wickets="HITTING"



    return {


        "pitching":"IN LINE",

        "impact":"IN LINE",

        "wickets":wickets,

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

        },


        "points":points

    }





# =====================================================
# ROUTES
# =====================================================


@app.route("/")
def index():

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



    video=request.files["video"]



    filename=(
        uuid.uuid4().hex+
        os.path.splitext(
            secure_filename(video.filename)
        )[1]
    )



    filepath=os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    video.save(filepath)




    points=detect_ball(
        filepath
    )



    cap=cv2.VideoCapture(
        filepath
    )


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



    result=analyse_lbw(
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
