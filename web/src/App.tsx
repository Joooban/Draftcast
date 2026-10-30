import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import Footer from './components/Footer'
import Insights from './pages/Insights'
import Method from './pages/Method'
import Predict from './pages/Predict'

export default function App() {
  return (
    <BrowserRouter>
      <header className="nav">
        <span className="brand">LoL Win Predictor</span>
        <nav>
          <NavLink to="/" end>
            Predict
          </NavLink>
          <NavLink to="/insights">Insights</NavLink>
          <NavLink to="/method">Method</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/insights" element={<Insights />} />
          <Route path="/method" element={<Method />} />
        </Routes>
      </main>
      <Footer />
    </BrowserRouter>
  )
}
