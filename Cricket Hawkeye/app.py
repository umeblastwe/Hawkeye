import os
import cv2
import uuid
import numpy as np

from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
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
# SERVE UPLOADED VIDEOS
# =====================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )



# =====================================================
# BALL TRACKING
# =====================================================

def detect_ball(video_path):

    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():

        return []


    points=[]

    previous_gray=None
    previous_point=None

    frame_id=0



    while True:


        ret,frame=cap.read()


        if not ret:
            break


        frame_id+=1


        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        gray=cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )



        hsv=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )



        # bright ball filter

        lower=np.array(
            [0,0,170]
        )

        upper=np.array(
            [180,80,255]
        )


        mask=cv2.inRange(
            hsv,
            lower,
            upper
        )



        # motion filter

        if previous_gray is not None:


            diff=cv2.absdiff(
                previous_gray,
                gray
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


        best=None
        best_area=0



        for c in contours:


            area=cv2.contourArea(c)


            if area < 3 or area > 150:
                continue



            x,y,w,h=cv2.boundingRect(c)



            ratio=w/h if h else 0


            if ratio <0.4 or ratio>2.5:
                continue



            cx=x+w//2
            cy=y+h//2



            if previous_point:


                distance=np.sqrt(
                    (cx-previous_point[0])**2+
                    (cy-previous_point[1])**2
                )


                if distance>120:
                    continue



            if area>best_area:

                best_area=area

                best=(cx,cy)




        if best:


            points.append({

                "frame":frame_id,
                "x":best[0],
                "y":best[1]

            })


            previous_point=best



        previous_gray=gray



    cap.release()


    return remove_outliers(points)





# =====================================================
# CLEAN WRONG POINTS
# =====================================================

def remove_outliers(points):


    if len(points)<3:

        return points



    clean=[points[0]]



    for p in points[1:]:


        last=clean[-1]


        d=np.sqrt(

            (p["x"]-last["x"])**2+
            (p["y"]-last["y"])**2

        )



        if d<150:

            clean.append(p)



    return clean





# =====================================================
# LBW ANALYSIS
# =====================================================

def analyse(points,width,height):


    if len(points)<5:


        return {

            "pitching":"UNKNOWN",

            "impact":"UNKNOWN",

            "wickets":"UNKNOWN",

            "decision":"NOT OUT"

        }



    xs=np.array(
        [p["x"] for p in points],
        dtype=float
    )


    ys=np.array(
        [p["y"] for p in points],
        dtype=float
    )



    bounce=np.argmax(
        ys
    )


    pitch={

        "x":float(xs[bounce]),

        "y":float(ys[bounce])

    }



    impact_index=min(
        bounce+4,
        len(xs)-1
    )


    impact={

        "x":float(xs[impact_index]),

        "y":float(ys[impact_index])

    }




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



    predicted_x=float(
        np.polyval(
            curve,
            stump_y
        )
    )



    projection={

        "x":predicted_x,

        "y":stump_y

    }



    left=width*0.46

    right=width*0.54




    if left<=predicted_x<=right:

        decision="OUT"

        wickets="HITTING"


    else:

        decision="NOT OUT"

        wickets="MISSING"



    return {


        "pitching":"IN LINE",

        "impact":"IN LINE",

        "wickets":wickets,

        "decision":decision,


        "pitch":pitch,

        "impact_point":impact,

        "projection":projection


    }





# =====================================================
# PAGES
# =====================================================

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


        return jsonify({

            "success":False,

            "error":"No video"

        })



    file=request.files["video"]



    filename=(
        uuid.uuid4().hex+
        os.path.splitext(
            secure_filename(file.filename)
        )[1]
    )



    path=os.path.join(
        UPLOAD_FOLDER,
        filename
    )



    file.save(path)




    points=detect_ball(
        path
    )



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




    result=analyse(
        points,
        width,
        height
    )




    return jsonify({

        "success":True,


        "video":url_for(
            "uploaded_file",
            filename=filename
        ),


        "ball_points":points,


        "analysis":result


    })




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
