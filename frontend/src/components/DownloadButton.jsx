function DownloadButton({ result }) {

    if (!result) return null;

    return (

        <div
            style={{
                marginTop: "30px"
            }}
        >

            <a
                href="http://127.0.0.1:8000/download"
            >

                <button>

                    Download Predicted CSV

                </button>

            </a>

        </div>

    );

}

export default DownloadButton;