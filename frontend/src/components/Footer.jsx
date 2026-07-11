function Footer() {
    return (
        <footer>
            <p>
                System Status: <span>Secure</span> &bull; Security Node: <span>Active</span> &bull; Telemetry: <span>Online</span>
            </p>
            <p style={{ fontSize: "0.75rem", opacity: 0.6, marginTop: "6px" }}>
                Model: <span>Random Forest</span> &bull; 
                Dataset: <span>CICIDS2017</span> &bull; 
                Backend: <span>FastAPI</span> &bull; 
                Frontend: <span>React</span> &bull; 
                Database: <span>SQLite</span> &bull; 
                Developer: <span>Ayush Singh</span>
            </p>
        </footer>
    );
}

export default Footer;