const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('public')); // To serve frontend static files if necessary

// Sample route
app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.get('/api/events', (req, res) => {
    res.json({ message: 'Event data' });
  });
  

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});