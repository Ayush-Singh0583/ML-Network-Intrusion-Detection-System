function SummaryCards({ result }) {

    if (!result) return null;

    return (

        <div>

            <h2>Prediction Summary</h2>

            <div className="cards">

                <div className="stat-card">

                    <h3>Total Records</h3>

                    <h1>{result.total_records}</h1>

                </div>

                {

                    Object.entries(result.summary).map(

                        ([attack, count]) => (

                            <div
                                key={attack}
                                className="stat-card"
                            >

                                <h3>{attack}</h3>

                                <h1>{count}</h1>

                            </div>

                        )

                    )

                }

            </div>

        </div>

    );

}

export default SummaryCards;