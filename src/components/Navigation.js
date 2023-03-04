import React from "react";
import { NavLink } from "react-router-dom";

function Navigation() {
  return (
    <nav>
      <h2>
        <a href="#home">AutoVision</a>
      </h2>
      <ul>
        <li>
          <a href="#section1">section1</a>
        </li>
        <li>
          <a href="#section2">section2</a>
        </li>
        <li>
          <a href="#section3">section3</a>
        </li>
        <li>
          <a href="#section4">section4</a>
        </li>
      </ul>
    </nav>
  );
}

export default Navigation;
