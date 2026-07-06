function UploadCard({ setFile, uploadFile, loading }) {

    return (

        <div className="card">

            <h2>Upload Network Traffic CSV</h2>

            <input
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files[0])}
            />

            <br />
            <br />

            <button
                onClick={uploadFile}
            >

                {loading ? "Predicting..." : "Predict"}

            </button>

        </div>

    );

}

export default UploadCard;