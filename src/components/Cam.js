import React, { useState } from "react";
const Cam = () => {
  return (
    <section id="home">
      <div className="video-container">
        <h1>See AutoVision in action!</h1>
        <video width="1280" height="720" controls>
          <source
            src="https://raw.githubusercontent.com/vincetrain/yrhacks-car-camera/main/car-passing-by.mp4"
            type="video/mp4"
          />
        </video>
      </div>
    </section>
  );
};
export default Cam;
