import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { UserService } from '../../user.service';
import { Router } from '@angular/router';

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

  constructor(public userService: UserService,
    private router: Router
  ) {
    for (let year = this.endYear; year >= this.startYear; year--) {
      this.years.push(year);
    }
  }

  onSubmit() {
    // Handle form submission here (e.g., send search data to a service)
    const isbn = (<HTMLInputElement>document.getElementById('isbn')).value;
    const title = (<HTMLInputElement>document.getElementById('title')).value;
    const author = (<HTMLInputElement>document.getElementById('author')).value;
    const publisher = (<HTMLInputElement>document.getElementById('publisher')).value;
    const startYear = (<HTMLSelectElement>document.getElementById('startYear')).value;
    const endYear = (<HTMLSelectElement>document.getElementById('endYear')).value;

    this.router.navigate(['/search'], {
      queryParams: {
        category: this.selectedCategory,
        isbn: isbn,
        title: title,
        author: author,
        publisher: publisher,
        startYear: startYear,
        endYear: endYear
      }
    });
  }
}
