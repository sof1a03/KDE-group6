import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { UserService } from '../../user.service';
import { Router } from '@angular/router';
import { CategoryService } from '../../category.service';

@Component({
  selector: 'app-search-menu',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './search-menu.component.html',
  styleUrl: './search-menu.component.css'
})
export class SearchMenuComponent {
  categories = ['Fiction', 'Non-Fiction', 'Mystery', 'Thriller', 'Sci-Fi', 'Romance', 'History'];
  selectedCategories = [];

  startYear = 1900;
  endYear = new Date().getFullYear();
  years: number[] = [];

  constructor(public userService: UserService,
    private router: Router,
    private categoryService: CategoryService
  ) {
    for (let year = this.endYear; year >= this.startYear; year--) {
      this.years.push(year);
    }
    categoryService.getCategories().subscribe(categories => {
      this.categories = categories;
    })
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
        categories: this.selectedCategories,
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
