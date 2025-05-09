package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// FluidFlowSimulation holds all simulation data
type FluidFlowSimulation struct {
	// Grid dimensions
	nx, ny int
	// Simulation parameters
	vInf    float64
	maxIter int
	tol     float64

	// Domain setup
	x, y     []float64
	dx, dy   float64
	X, Y     [][]float64
	psi, phi [][]float64
	mask     [][]bool
	u, v     [][]float64
	p        [][]float64
	cylinder *Circle
	airfoil  *Airfoil
}

// Circle represents a circular object in the flow
type Circle struct {
	centerX, centerY float64
	radius           float64
}

// Airfoil represents an airfoil shape in the flow
type Airfoil struct {
	centerX, centerY      float64
	chord, thicknessRatio float64
	camber, camberPos     float64
}

// NewFluidFlowSimulation creates a new fluid flow simulation
func NewFluidFlowSimulation(nx, ny int, vInf float64, maxIter int, tol float64) *FluidFlowSimulation {
	sim := &FluidFlowSimulation{
		nx:      nx,
		ny:      ny,
		vInf:    vInf,
		maxIter: maxIter,
		tol:     tol,
	}

	// Setup domain
	sim.x = linspace(0, 10, nx)
	sim.y = linspace(-5, 5, ny)
	sim.dx = sim.x[1] - sim.x[0]
	sim.dy = sim.y[1] - sim.y[0]

	// Create 2D grids
	sim.X = make([][]float64, ny)
	sim.Y = make([][]float64, ny)
	sim.psi = make([][]float64, ny)
	sim.phi = make([][]float64, ny)
	sim.mask = make([][]bool, ny)
	sim.u = make([][]float64, ny)
	sim.v = make([][]float64, ny)
	sim.p = make([][]float64, ny)

	for j := 0; j < ny; j++ {
		sim.X[j] = make([]float64, nx)
		sim.Y[j] = make([]float64, nx)
		sim.psi[j] = make([]float64, nx)
		sim.phi[j] = make([]float64, nx)
		sim.mask[j] = make([]bool, nx)
		sim.u[j] = make([]float64, nx)
		sim.v[j] = make([]float64, nx)
		sim.p[j] = make([]float64, nx)

		// Initialize mask as all fluid
		for i := 0; i < nx; i++ {
			sim.mask[j][i] = true
		}

		// Create meshgrid
		for i := 0; i < nx; i++ {
			sim.X[j][i] = sim.x[i]
			sim.Y[j][i] = sim.y[j]
		}
	}

	// Set boundary conditions
	sim.setBoundaryConditions()

	return sim
}

// setBoundaryConditions sets initial boundary conditions for stream function
func (sim *FluidFlowSimulation) setBoundaryConditions() {
	// Inlet BC (x=0): u = v_inf, v = 0
	// Stream function varies linearly with y
	for j := 0; j < sim.ny; j++ {
		sim.psi[j][0] = sim.vInf * sim.y[j]
	}

	// Top boundary (y=ymax): constant psi
	for i := 0; i < sim.nx; i++ {
		sim.psi[0][i] = sim.vInf * sim.y[0]
	}

	// Bottom boundary (y=ymin): constant psi
	for i := 0; i < sim.nx; i++ {
		sim.psi[sim.ny-1][i] = sim.vInf * sim.y[sim.ny-1]
	}

	// Outlet boundary (x=xmax): zero gradient
	for j := 0; j < sim.ny; j++ {
		sim.psi[j][sim.nx-1] = sim.psi[j][sim.nx-2]
	}

	// Initialize velocity potential (phi) - uniform flow in x direction
	for i := 0; i < sim.nx; i++ {
		for j := 0; j < sim.ny; j++ {
			sim.phi[j][i] = sim.vInf * sim.x[i]
		}
	}
}

// AddCylinder adds a circular object to the flow field
func (sim *FluidFlowSimulation) AddCylinder(centerX, centerY, radius float64) {
	// Mask solid points
	for i := 0; i < sim.nx; i++ {
		for j := 0; j < sim.ny; j++ {
			if math.Pow(sim.x[i]-centerX, 2)+math.Pow(sim.y[j]-centerY, 2) <= math.Pow(radius, 2) {
				sim.mask[j][i] = false // Mark as solid
			}
		}
	}

	// Store for visualization
	sim.cylinder = &Circle{
		centerX: centerX,
		centerY: centerY,
		radius:  radius,
	}
}

// AddAirfoil adds a cambered NACA 4-digit airfoil to the mask
func (sim *FluidFlowSimulation) AddAirfoil(centerX, centerY, chord, thicknessRatio, camber, camberPos float64) {
	for i := 0; i < sim.nx; i++ {
		for j := 0; j < sim.ny; j++ {
			xAbs := sim.x[i]
			yAbs := sim.y[j]

			xRel := xAbs - centerX
			yRel := yAbs - centerY

			if xRel >= 0 && xRel <= chord {
				xNorm := xRel / chord

				// Camber line z_c and its slope dzc_dx
				var zC, dzcDx float64
				if xNorm < camberPos {
					zC = (camber / math.Pow(camberPos, 2)) * (2*camberPos*xNorm - math.Pow(xNorm, 2))
					dzcDx = (2 * camber / math.Pow(camberPos, 2)) * (camberPos - xNorm)
				} else {
					zC = (camber / math.Pow(1-camberPos, 2)) * ((1 - 2*camberPos) + 2*camberPos*xNorm - math.Pow(xNorm, 2))
					dzcDx = (2 * camber / math.Pow(1-camberPos, 2)) * (camberPos - xNorm)
				}

				theta := math.Atan(dzcDx)

				// Thickness distribution
				yt := 5 * thicknessRatio * chord * (0.2969*math.Sqrt(xNorm) -
					0.1260*xNorm -
					0.3516*math.Pow(xNorm, 2) +
					0.2843*math.Pow(xNorm, 3) -
					0.1015*math.Pow(xNorm, 4))

				// Upper and lower surface positions
				yUpper := zC + yt*math.Cos(theta)
				yLower := zC - yt*math.Cos(theta)

				if yRel >= yLower && yRel <= yUpper {
					sim.mask[j][i] = false
				}
			}
		}
	}

	sim.airfoil = &Airfoil{
		centerX:        centerX,
		centerY:        centerY,
		chord:          chord,
		thicknessRatio: thicknessRatio,
		camber:         camber,
		camberPos:      camberPos,
	}
}

// SolveStreamFunction solves the Laplace equation for the stream function
func (sim *FluidFlowSimulation) SolveStreamFunction() {
	psiOld := make([][]float64, sim.ny)
	for j := 0; j < sim.ny; j++ {
		psiOld[j] = make([]float64, sim.nx)
		copy(psiOld[j], sim.psi[j])
	}

	// Use multiple goroutines for parallelization
	numCPU := runtime.NumCPU()

	for iterCount := 0; iterCount < sim.maxIter; iterCount++ {
		// Use parallel processing for the interior points
		var wg sync.WaitGroup
		rowsPerGoroutine := (sim.ny - 2) / numCPU
		if rowsPerGoroutine < 1 {
			rowsPerGoroutine = 1
		}

		for cpu := 0; cpu < numCPU; cpu++ {
			wg.Add(1)
			startRow := 1 + cpu*rowsPerGoroutine
			endRow := startRow + rowsPerGoroutine
			if cpu == numCPU-1 {
				endRow = sim.ny - 1 // ensure we cover all rows
			}
			if endRow > sim.ny-1 {
				endRow = sim.ny - 1
			}

			go func(startRow, endRow int) {
				defer wg.Done()
				// Gauss-Seidel iteration for interior points
				for j := startRow; j < endRow; j++ {
					for i := 1; i < sim.nx-1; i++ {
						if sim.mask[j][i] {
							sim.psi[j][i] = 0.25 * (sim.psi[j][i+1] + sim.psi[j][i-1] +
								sim.psi[j+1][i] + sim.psi[j-1][i])
						}
					}
				}
			}(startRow, endRow)
		}
		wg.Wait()

		// Outlet boundary condition (zero gradient)
		for j := 0; j < sim.ny; j++ {
			sim.psi[j][sim.nx-1] = sim.psi[j][sim.nx-2]
		}

		// Check convergence
		maxDiff := 0.0
		for j := 0; j < sim.ny; j++ {
			for i := 0; i < sim.nx; i++ {
				diff := math.Abs(sim.psi[j][i] - psiOld[j][i])
				if diff > maxDiff {
					maxDiff = diff
				}
				psiOld[j][i] = sim.psi[j][i]
			}
		}

		if maxDiff < sim.tol {
			fmt.Printf("Stream function converged after %d iterations\n", iterCount)
			break
		}
	}

	// Set inlet velocity
	for j := 0; j < sim.ny; j++ {
		sim.u[j][0] = sim.vInf
		sim.v[j][0] = 0
	}

	// Calculate velocity field
	sim.calculateVelocityField()
}

// calculateVelocityField calculates velocity from stream function using central differences
func (sim *FluidFlowSimulation) calculateVelocityField() {
	// Use goroutines for parallel processing
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	rowsPerGoroutine := (sim.ny - 2) / numCPU
	if rowsPerGoroutine < 1 {
		rowsPerGoroutine = 1
	}

	for cpu := 0; cpu < numCPU; cpu++ {
		wg.Add(1)
		startRow := 1 + cpu*rowsPerGoroutine
		endRow := startRow + rowsPerGoroutine
		if cpu == numCPU-1 {
			endRow = sim.ny - 1 // ensure we cover all rows
		}
		if endRow > sim.ny-1 {
			endRow = sim.ny - 1
		}

		go func(startRow, endRow int) {
			defer wg.Done()
			// Calculate velocity for interior points
			for j := startRow; j < endRow; j++ {
				for i := 1; i < sim.nx-1; i++ {
					if sim.mask[j][i] {
						// u = dpsi/dy
						sim.u[j][i] = (sim.psi[j+1][i] - sim.psi[j-1][i]) / (2 * sim.dy)
						// v = -dpsi/dx
						sim.v[j][i] = -(sim.psi[j][i+1] - sim.psi[j][i-1]) / (2 * sim.dx)
					}
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()

	// One-sided differences for walls
	for j := 1; j < sim.ny-1; j++ {
		if sim.mask[j][sim.nx-1] {
			// Forward difference for u
			sim.u[j][sim.nx-1] = sim.u[j][sim.nx-2]
			// Backward difference for v
			sim.v[j][sim.nx-1] = -(sim.psi[j][sim.nx-1] - sim.psi[j][sim.nx-2]) / sim.dx
		}
	}

	// Calculate pressure distribution using Bernoulli's equation
	vInfSquared := sim.vInf * sim.vInf
	pInf := 0.0 // reference pressure
	rho := 1.0  // density (assumed constant)

	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			vSquared := sim.u[j][i]*sim.u[j][i] + sim.v[j][i]*sim.v[j][i]
			// P/ρ + V²/2 = P_inf/ρ + V_inf²/2
			sim.p[j][i] = pInf + 0.5*rho*(vInfSquared-vSquared)
		}
	}
}

// SaveResults saves plots of simulation results
func (sim *FluidFlowSimulation) SaveResults() {
	// Stream function plot
	sim.savePlot("stream_function", sim.psi, "Stream Function")

	// Velocity magnitude
	velMag := make([][]float64, sim.ny)
	for j := 0; j < sim.ny; j++ {
		velMag[j] = make([]float64, sim.nx)
		for i := 0; i < sim.nx; i++ {
			velMag[j][i] = math.Sqrt(sim.u[j][i]*sim.u[j][i] + sim.v[j][i]*sim.v[j][i])
		}
	}
	sim.savePlot("velocity_magnitude", velMag, "Velocity Magnitude")

	// Pressure plot
	sim.savePlot("pressure", sim.p, "Pressure Field")

	// Create velocity field quiver plot
	sim.saveVelocityPlot("velocity_field")
}

// savePlot saves a contour plot for a given field
func (sim *FluidFlowSimulation) savePlot(filename string, field [][]float64, title string) {

	// Save data to a text file instead (simple alternative to heatmap)

	dataDir := "data"
	filepath := filepath.Join(dataDir, filename+".data")

	f, err := os.Create(filepath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Write field data
	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			fmt.Fprintf(f, "%f %f %f\n", sim.x[i], sim.y[j], field[j][i])
		}
		fmt.Fprintln(f)
	}

	fmt.Printf("Data for %s saved to %s.data\n", title, filename)
}

// saveVelocityPlot saves vectors to a data file
func (sim *FluidFlowSimulation) saveVelocityPlot(filename string) {
	dataDir := "data"
	filepath := filepath.Join(dataDir, filename+".data")

	f, err := os.Create(filepath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Write velocity vector data
	skip := 5 // Skip some points for better visualization
	for j := 0; j < sim.ny; j += skip {
		for i := 0; i < sim.nx; i += skip {
			if sim.mask[j][i] {
				fmt.Fprintf(f, "%f %f %f %f\n",
					sim.X[j][i], sim.Y[j][i],
					sim.u[j][i], sim.v[j][i])
			}
		}
	}

	fmt.Printf("Velocity data saved to %s.data\n", filename)
}

// Utility functions
func linspace(start, end float64, num int) []float64 {
	result := make([]float64, num)
	step := (end - start) / float64(num-1)
	for i := range result {
		result[i] = start + float64(i)*step
	}
	return result
}

func scale(x, inMin, inMax, outMin, outMax float64) float64 {
	return (x-inMin)*(outMax-outMin)/(inMax-inMin) + outMin
}

func findIndex(array []float64, value float64) int {
	for i := 0; i < len(array)-1; i++ {
		if value >= array[i] && value < array[i+1] {
			return i
		}
	}
	return len(array) - 1
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func randomBetween(min, max float64) float64 {
	return min + (max-min)*rand.Float64()
}

// Plot implements the plot.Plotter interface for drawing a circle
func (c *Circle) Plot(canvas draw.Canvas, plt *plot.Plot) {
	// Simple implementation - just draw a point for now
	pts := plotter.XYs{{X: c.centerX, Y: c.centerY}}
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Color = color.RGBA{R: 0, G: 0, B: 0, A: 255}
	scatter.GlyphStyle.Radius = vg.Points(5)
	scatter.Plot(canvas, plt)
}

func main() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	// Create simulation
	sim := NewFluidFlowSimulation(100, 100, 1.0, 30000, 1e-6)

	// Uncomment the desired object
	//sim.AddCylinder(5.0, 0.0, 1.0)
	sim.AddAirfoil(2.5, 0.0, 7.0, 0.12, 0.02, 0.4)

	// Solve and visualize
	sim.SolveStreamFunction()
	sim.SaveResults()

	fmt.Println("Simulation completed. Results saved to image files and flow_animation.gif")
}
