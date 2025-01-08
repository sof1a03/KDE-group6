import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-search-menu',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './search-menu.component.html',
  styleUrl: './search-menu.component.css'
})
export class SearchMenuComponent {
  categories = ['Fiction', 'Non-Fiction', 'Mystery', 'Thriller', 'Sci-Fi', 'Romance', 'History'];
  selectedCategory = this.categories[0];

  startYear = 1900;
  endYear = new Date().getFullYear();
  years: number[] = [];

  constructor() {
    for (let year = this.endYear; year >= this.startYear; year--) {
      this.years.push(year);
    }
  }

  onSubmit() {
    // Handle form submission here (e.g., send search data to a service)
    console.log('Search submitted!');
  }
}
