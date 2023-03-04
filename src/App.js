import React from "react";
import Navigation from "./components/Navigation";
import Cam from "./components/Cam";
import "./App.css";
import SmoothScroll from "smooth-scroll";

export const scroll = new SmoothScroll('a[href*="#"]', {
  speed: 500,
  speedAsDuration: true
});

function App() {
  return (
    <div>
      <Navigation />
      <Cam />
    </div>
  );
}

export default App;
