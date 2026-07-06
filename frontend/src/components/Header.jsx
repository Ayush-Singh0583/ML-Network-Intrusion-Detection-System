import { FaShieldAlt } from "react-icons/fa";

function Header() {
    return (
        <div className="header">

            <h1>
                <FaShieldAlt />
                {" "}
                Network Intrusion Detection System
            </h1>

            <p>
                Random Forest Model | CIC IDS 2017 Dataset
            </p>

        </div>
    );
}

export default Header;